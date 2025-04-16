import logging
import os
import uuid
import time
import json
from typing import Dict, List, Any, Optional
import asyncio
import dotenv
from datetime import datetime
import re
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils.metrics import (
    increment_counter, record_time, record_api_call, flush_metrics,
    record_feedback,record_safety_score, record_empathy_score, 
    record_memory_usage, record_measurement
)
# In app.py, after other utils imports
try:
    from utils.response_evaluation import check_response_safety, calculate_empathy
    EVALUATION_FUNCTIONS_LOADED = True
except ImportError:
    logger.error("Could not import evaluation functions from utils.response_evaluation. Placeholders will be used.")
    EVALUATION_FUNCTIONS_LOADED = False

    # Define placeholders directly in app.py if import fails
    def check_response_safety(text: str) -> bool:
        logger.warning("Using placeholder safety check (always True)")
        return True

    def calculate_empathy(text: str) -> float:
        logger.warning("Using placeholder empathy calculation (always 0.5)")
        return 0.5
    
from datetime import datetime, timedelta # Import datetime stuff
from fastapi import APIRouter, Depends, HTTPException, Query # If using router, else adapt

# Set up environment variables
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our classes
from chatbot.multi_aspect_processor import MultiAspectQueryProcessor
from chatbot.memory_manager import MemoryManager
from utils.location_handler import (
    extract_location_from_input,
    get_abortion_access_info,
    generate_abortion_access_response
)

# Create FastAPI app
app = FastAPI(title="Abby Chatbot API", 
              description="Reproductive health chatbot with multi-aspect query handling",
              version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/css", StaticFiles(directory="static/css"), name="css")
app.mount("/js", StaticFiles(directory="static/js"), name="js") 

# Configure templates
templates = Jinja2Templates(directory="templates")

# Initialize global components
memory_manager = MemoryManager()

# Check for OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("⚠️ OPENAI_API_KEY environment variable not found. Response formatting improvements will be disabled.")
    logger.warning("To enable OpenAI response formatting, set the OPENAI_API_KEY environment variable.")
else:
    logger.info("OpenAI API key found - response formatting improvements enabled.")

# Check for serialized models
serialized_models_path = Path('serialized_models')
if serialized_models_path.exists() and os.path.isfile('load_serialized_models.py'):
    logger.info('Serialized models found, attempting to load...')
    try:
        import load_serialized_models
        if load_serialized_models.check_serialized_models():
            logger.info('Using serialized models for faster initialization')
            # Load serialized models into global components
            loaded_models = load_serialized_models.load_all_models()
            # Memory manager and query processor will still be initialized normally
        else:
            logger.info('Serialized models check failed, using normal initialization')
    except Exception as e:
        logger.error(f'Error loading serialized models: {str(e)}')
        logger.info('Falling back to normal initialization')
else:
    logger.info('No serialized models found, using normal initialization')

query_processor = MultiAspectQueryProcessor()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_location: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Location-based query flags
    is_location_clinic_query: Optional[bool] = False
    is_location_policy_query: Optional[bool] = False
    zip_code: Optional[str] = None
    city_name: Optional[str] = None
    state_name: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    message_id: str
    session_id: str
    citations: List[Any] = []
    citation_objects: List[Any] = []
    timestamp: float
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    graphics: Optional[List[Dict[str, Any]]] = None
    show_map: Optional[bool] = None
    zip_code: Optional[str] = None
    # Fields for abortion access info
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    is_legal: Optional[bool] = None
    # Fields for travel recommendations when abortion is banned
    travel_state_code: Optional[str] = None
    travel_state_name: Optional[str] = None
    requires_travel: Optional[bool] = None
    nearby_states: Optional[List[str]] = None
    map_data: Optional[Dict[str, Any]] = None

class SessionRequest(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class FeedbackRequest(BaseModel):
    message_id: str
    session_id: Optional[str] = None # Add if needed and sent from frontend
    rating: int
    comment: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None # Keep if you plan to send these

# Dependency to get the processor
def get_processor():
    return query_processor

# Dependency to get the memory manager
def get_memory_manager():
    return memory_manager

# Main page route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main chat interface
    """
    # Get the Google Maps API key from environment variables
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    
    # Log the key status (not the actual key)
    if google_maps_api_key:
        logger.info("Google Maps API key is set")
    else:
        logger.warning("Google Maps API key is not set - map functionality will be limited")
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "google_maps_api_key": google_maps_api_key
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, 
               processor: MultiAspectQueryProcessor = Depends(get_processor),
               memory: MemoryManager = Depends(get_memory_manager),
               debug: bool = Query(False, description="Enable debug mode")):
    """
    Main chat endpoint that processes user messages, maintains history
    and returns appropriate responses.
    """
    start_time = time.time()
    
    # Generate message_id early to avoid UnboundLocalError
    message_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4()) # Ensure session_id is defined early

    try:
        logger.info(f"Received chat request: {request.message[:100]}")
        
        # Track message received
        increment_counter('user_messages')

        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        # Optional: Log message length as a separate event if useful
        record_measurement('user_message_length', len(request.message), session_id=session_id)

        # New session start event
        if not request.session_id:
            increment_counter('session_starts', session_id=session_id) # Log session_id with start

        # Store user message in memory
        memory.add_message(
            session_id=session_id,
            message=request.message,
            role="user",
            metadata=request.metadata
        )
        
        # Get conversation history
        conversation_history = memory.get_history(session_id)
        
        # Process the query through our multi-aspect processor
        query_start_time = time.time()
        
        # --- MODIFICATION START ---
        effective_query = request.message # Default to user's message
        original_intent = None # Track intent if location provided standalone

        # Check if the previous bot message asked for location
        needs_location_context = False
        if conversation_history and len(conversation_history) > 1:
            # Check the second to last message (last bot message)
            last_bot_msg_index = -1
            if conversation_history[-1]['role'] == 'user':
                last_bot_msg_index = -2
            if len(conversation_history) >= abs(last_bot_msg_index):
                 last_bot_message = conversation_history[last_bot_msg_index]
                 if last_bot_message.get('role') == 'assistant' and \
                    last_bot_message.get('metadata', {}).get('needs_state_info'):
                         needs_location_context = True
                         logger.info("Detected user is providing location after being prompted.")
        
        # Check for direct ZIP code input
        zip_match = re.search(r'\b(\d{5})\b', request.message)
        state_match_found = False
        if zip_match and len(request.message.strip()) <= 10:  # If the message is just a ZIP code
            zip_code = zip_match.group(1)
            logger.info(f"Detected direct ZIP code input: {zip_code}")
            
            # Import preprocessor to get state from ZIP
            from chatbot.preprocessor import Preprocessor
            preprocessor = Preprocessor()
            
            # Get state from ZIP
            state = preprocessor.get_state_from_zip(zip_code)
            if state:
                logger.info(f"Resolved ZIP code {zip_code} to state {state}")
                
                # Set user location for the processor
                if not request.user_location:
                    request.user_location = {}
                request.user_location["zip_code"] = zip_code
                request.user_location["state"] = state
                
                # Mark as a location query
                request.is_location_policy_query = True
                
                # CONSTRUCT EFFECTIVE QUERY if location provided after prompt
                if needs_location_context:
                    effective_query = f"Check abortion policy and clinics for {state} (ZIP: {zip_code})"
                    logger.info(f"Constructed effective query: {effective_query}")
                state_match_found = True
        
        # Check for direct state name input (e.g., "Texas" or "TX")
        elif not zip_match and len(request.message.strip()) <= 20:  # Short message that's not a ZIP code
            # State name lookup
            state_names = {
                "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", 
                "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
                "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
                "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
                "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
                "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
                "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
                "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
                "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
                "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
                "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
                "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
                "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
            }
            
            # Check for full state name
            input_text = request.message.strip()
            state_match_found = False
            matched_state_name = None
            matched_state_code = None
            
            for state_name, state_code in state_names.items():
                # Case insensitive match for full state name
                if re.search(r'\b' + re.escape(state_name) + r'\b', input_text, re.IGNORECASE):
                    logger.info(f"Detected direct state name input: {state_name}")
                    
                    # Set user location for the processor
                    if not request.user_location:
                        request.user_location = {}
                    request.user_location["state"] = state_name
                    request.user_location["state_code"] = state_code
                    
                    # Mark as a location query
                    request.is_location_policy_query = True
                    matched_state_name = state_name
                    matched_state_code = state_code
                    state_match_found = True
                    
                    # CONSTRUCT EFFECTIVE QUERY if location provided after prompt
                    if needs_location_context:
                        effective_query = f"Check abortion policy for {matched_state_name}"
                        logger.info(f"Constructed effective query: {effective_query}")
                    break
                
                # Check for state abbreviation
                if re.search(r'\b' + re.escape(state_code) + r'\b', input_text, re.IGNORECASE):
                    logger.info(f"Detected direct state code input: {state_code}")
                    
                    # Set user location for the processor
                    if not request.user_location:
                        request.user_location = {}
                    request.user_location["state"] = state_name
                    request.user_location["state_code"] = state_code
                    
                    # Mark as a location query
                    request.is_location_policy_query = True
                    matched_state_name = state_name
                    matched_state_code = state_code
                    state_match_found = True
                    
                    # CONSTRUCT EFFECTIVE QUERY if location provided after prompt
                    if needs_location_context:
                        effective_query = f"Check abortion policy for {matched_state_name}"
                        logger.info(f"Constructed effective query: {effective_query}")
                    break
        
        # Check for vague location queries next
        message_lower = request.message.lower()
        vague_location_patterns = [
            r'\b(?:my|our)\s+state\b',
            r'\bin\s+my\s+(?:area|region|location|city|town)\b',
            r'\bnear\s+(?:me|us|here)\b',
            r'\b(?:here|nearby|locally)\b',
            r'\bwhere\s+(?:i|we)\s+(?:am|are|live|stay)\b'
        ]
        
        is_vague_location_query = any(re.search(pattern, message_lower) for pattern in vague_location_patterns) and (
            "abortion" in message_lower or 
            "access" in message_lower or 
            "legal" in message_lower or 
            "clinic" in message_lower
        )
        
        if is_vague_location_query and not needs_location_context:
            logger.info("Detected vague location query about abortion access")
            
            # Skip full processing and just send a clarification prompt
            clarification_text = "Could you please share your ZIP code or state so I can give you accurate information about abortion access in your area?"
            
            response_data = {
                "text": clarification_text,
                "message_id": message_id,
                "session_id": session_id,
                "needs_state_info": True,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
                "aspect_type": "policy"
            }
            
            # Store the bot's response in memory
            memory.add_message(
                session_id=session_id, 
                message=clarification_text,
                role="assistant",
                metadata={"message_id": message_id, "needs_state_info": True} # Add metadata to bot message too
            )
            
            return response_data
        
        # Continue with normal processing
        logger.info(f"Processing effective query: {effective_query[:100]}...")
        response_data = await processor.process_query(
            message=effective_query, # Use effective query here
            original_message=request.message, # Pass original user input for reference/history
            conversation_history=conversation_history,
            user_location=request.user_location,
            session_id=session_id
        )
        query_processing_time = time.time() - query_start_time
        
        # Record query processing time
        record_time('query_processing', query_processing_time, session_id=session_id, category=response_data.get("aspect_type"))

        # Add message ID and session ID to response
        message_id = str(uuid.uuid4())
        response_data["message_id"] = message_id
        response_data["session_id"] = session_id

        if "processing_time" not in response_data:
            response_data["processing_time"] = time.time() - start_time
            
        # Fix response text if it's missing
        if "text" not in response_data and "primary_content" in response_data:
            logger.info("Using primary_content as text in response")
            response_data["text"] = response_data["primary_content"]
        
        # Clean up duplicated content in abortion policy responses
        if "text" in response_data and "abortion" in request.message.lower() and ("policy" in request.message.lower() or "law" in request.message.lower()):
            # Check if this is specifically about a state's abortion policy
            state_pattern = r"\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming|DC)\b"
            
            state_match = re.search(state_pattern, request.message, re.IGNORECASE)
            if state_match:
                state_name = state_match.group(1)
                logger.info(f"Cleaning up duplicated content for {state_name} abortion policy")
                
                # Apply text cleanup
                response_text = response_data["text"]
                
                # For banned states like Texas
                if f"Abortion is not currently legal in {state_name}" in response_text and f"Abortion is not currently legal in {state_name} except in" in response_text:
                    # Keep only the more detailed version
                    response_text = response_text.replace(f"Abortion is not currently legal in {state_name}.\n\n• Abortion is banned with very limited exceptions.\n\n", "")
                
                # For legal states (like California), remove redundant sections
                if f"Yes, abortion is legal in {state_name}" in response_text and f"Here's abortion policy information for {state_name}" in response_text:
                    # Extract the first section up to "You can see nearby clinics"
                    if "You can see nearby clinics" in response_text:
                        first_part = response_text.split("You can see nearby clinics")[0]
                        map_text = "You can see nearby clinics on the map below."
                        # Remove the redundant "Here's abortion policy information" section
                        if "Here's abortion policy information for" in response_text:
                            second_part = response_text.split("Here's abortion policy information for")[1]
                            if "." in second_part:
                                second_part = second_part.split(".", 1)[1]
                            response_text = first_part + map_text
                
                # If there's a duplicative section with "Detailed policy information for State:"
                if f"<strong>Detailed policy information for {state_name}:" in response_text and "• Abortion is banned" in response_text:
                    # Clean up the patterns to avoid redundancy
                    sections = response_text.split(f"<p class='message-paragraph'>")
                    if len(sections) > 1:
                        response_text = sections[0] + "\n<p class='message-paragraph'>" + sections[1]
                
                # Cleanup for redundant bullet points in California and other states
                if "<p class='message-paragraph'>Here's abortion policy information for" in response_text:
                    # Check if we have duplicate bullet lists
                    bullets_pattern = r"<ul class='chat-bullet-list'>.*?</ul>"
                    bullet_matches = re.findall(bullets_pattern, response_text, re.DOTALL)
                    
                    if len(bullet_matches) > 1:
                        # Keep only the first bullet list and remove others
                        for match in bullet_matches[1:]:
                            response_text = response_text.replace(match, "")
                
                # Check for repeating disclaimers about "information is based on the most recent data"
                disclaimer1 = "This information is based on the most recent data available, but laws may have changed."
                disclaimer2 = "This information is based on the most recent data available. For the most up-to-date information"
                
                if disclaimer1 in response_text and disclaimer2 in response_text:
                    # Keep only one disclaimer
                    response_text = response_text.replace(disclaimer1, "")
                
                response_data["text"] = response_text
        
        # Check if this is a location-based query about clinics or policy and enhance the response
        is_clinic_search = False
        is_policy_search = False
        clinic_keywords = ["clinic", "health center", "family planning", "planned parenthood", "medical center", "provider", "facility", "facilities", "doctor"]
        abortion_keywords = ["abortion"]
        policy_keywords = ["policy", "law", "legal", "restriction", "allowed", "banned", "permitted"]
        map_keywords = ["map", "show on map", "display", "locate", "find", "where"]
        
        # Simple dictionary to map zip code prefixes to states (first 3 digits)
        # This is a simplified approach - in a real system, you would use a more comprehensive ZIP code database
        ZIP_TO_STATE = {
            "941": "California", "942": "California", "943": "California", "944": "California", "945": "California", 
            "946": "California", "947": "California", "948": "California", "949": "California", "950": "California",
            "951": "California", "952": "California", "953": "California", "954": "California", "955": "California",
            "956": "California", "959": "California", "960": "California", "961": "California",
            "900": "California", "901": "California", "902": "California", "903": "California", "904": "California",
            "905": "California", "906": "California", "907": "California", "908": "California", "910": "California",
            "911": "California", "912": "California", "913": "California", "914": "California", "915": "California",
            "916": "California", "917": "California", "918": "California", "919": "California", "920": "California",
            "921": "California", "922": "California", "923": "California", "924": "California", "925": "California",
            "926": "California", "927": "California", "928": "California", "930": "California", "931": "California",
            "932": "California", "933": "California", "934": "California", "935": "California", "936": "California",
            "937": "California", "938": "California", "939": "California", "940": "California",
            "100": "New York", "101": "New York", "102": "New York", "103": "New York", "104": "New York",
            "105": "New York", "106": "New York", "107": "New York", "108": "New York", "109": "New York",
            "110": "New York", "111": "New York", "112": "New York", "113": "New York", "114": "New York",
            "115": "New York", "116": "New York", "117": "New York", "118": "New York", "119": "New York",
            "120": "New York", "121": "New York", "122": "New York", "123": "New York", "124": "New York",
            "125": "New York", "126": "New York", "127": "New York", "128": "New York", "129": "New York",
            "130": "New York", "131": "New York", "132": "New York", "133": "New York", "134": "New York",
            "135": "New York", "136": "New York", "137": "New York", "138": "New York", "139": "New York",
            "140": "New York", "141": "New York", "142": "New York", "143": "New York", "144": "New York",
            "145": "New York", "146": "New York", "147": "New York", "148": "New York", "149": "New York",
            "630": "Texas", "631": "Texas", "765": "Texas", "766": "Texas", "767": "Texas", "768": "Texas",
            "769": "Texas", "770": "Texas", "771": "Texas", "772": "Texas", "773": "Texas", "774": "Texas",
            "775": "Texas", "776": "Texas", "777": "Texas", "778": "Texas", "779": "Texas", "780": "Texas",
            "781": "Texas", "782": "Texas", "783": "Texas", "784": "Texas", "785": "Texas", "786": "Texas",
            "787": "Texas", "788": "Texas", "789": "Texas", "790": "Texas", "791": "Texas", "792": "Texas",
            "793": "Texas", "794": "Texas", "795": "Texas", "796": "Texas", "797": "Texas", "798": "Texas",
            "799": "Texas", "880": "Texas", "885": "Texas"
        }
        
        # Function to get state name from ZIP code
        def get_state_from_zip(zip_code):
            if not zip_code or len(zip_code) < 3:
                return None
                
            prefix = zip_code[:3]
            return ZIP_TO_STATE.get(prefix)
        
        # Lookup policy information by state
        POLICIES_BY_STATE = {
            "California": {
                "legal": True,
                "gestational_limit": "Abortion is legal throughout pregnancy if necessary to protect the life or health of the pregnant person.",
                "insurance_coverage": "State Medicaid and private insurance plans are required to cover abortion care.",
                "waiting_period": "No mandatory waiting period.",
                "summary": "California has strong protections for abortion access. Abortion is legal and protected by state law."
            },
            "New York": {
                "legal": True,
                "gestational_limit": "Abortion is legal up to 24 weeks of pregnancy, and after that point if necessary to protect the patient's life or health.",
                "insurance_coverage": "State Medicaid and many private insurance plans cover abortion care.",
                "waiting_period": "No mandatory waiting period.",
                "summary": "New York has strong protections for abortion access. Abortion is legal and protected by state law."
            },
            "Texas": {
                "legal": False,
                "gestational_limit": "Abortion is banned with very limited exceptions.",
                "insurance_coverage": "Most insurance plans are prohibited from covering abortion.",
                "waiting_period": "Not applicable due to ban.",
                "summary": "Texas has banned nearly all abortions. The ban allows exceptions only when necessary to prevent death or 'substantial impairment of major bodily function' of the pregnant person."
            }
        }
        
        # Enhanced location detection based on client-side processing
        # First check for location-based policy queries
        if request.is_location_policy_query:
            logger.info("Client detected location-based policy query")
            
            # Get the full text of the message for location extraction
            message_text = request.message
            
            # Check for "my state" queries and handle them immediately
            my_state_pattern = r'\b(?:my|our)\s+state\b'
            if re.search(my_state_pattern, message_text, re.IGNORECASE):
                logger.info("Detected 'my state' query, sending immediate clarification prompt")
                
                # Create an immediate clarification response
                response_data = {
                    "text": "Could you please share your ZIP code or state so I can give you accurate information about abortion access in your area?",
                    "message_id": message_id,
                    "session_id": session_id,
                    "needs_state_info": True,
                    "timestamp": time.time(),
                    "processing_time": time.time() - start_time,
                    "aspect_type": "policy"
                }
                
                # Store the bot's response in memory
                memory.add_message(
                    session_id=session_id, 
                    message=response_data["text"],
                    role="assistant",
                    metadata={"message_id": message_id}
                )
                
                return response_data
                
            # Import our ambiguous abbreviations list
            from chatbot.policy_handler import AMBIGUOUS_ABBREVIATIONS
            
            # Check for common ambiguous abbreviations being misinterpreted
            message_lower = message_text.lower()
            for abbr in AMBIGUOUS_ABBREVIATIONS:
                # Skip if it's a clear state mention in uppercase
                if abbr.upper() in message_text:
                    continue
                    
                # Check if it might be misinterpreted as a state
                if f" {abbr} " in f" {message_lower} ":
                    # Make sure it's not in a state context
                    state_context_patterns = [
                        rf'state:?\s*{re.escape(abbr)}',
                        rf'(?:in|from|to|at|live(?:s|d)? in|move(?:d|ing)? to|reside(?:s|d)? in)\s+{re.escape(abbr)}(?:\b|\s|$)',
                        rf'(?:abortion|laws|policy|policies) (?:in|for) {re.escape(abbr)}'
                    ]
                    has_state_context = any(re.search(pattern, message_text, re.IGNORECASE) for pattern in state_context_patterns)
                    
                    if not has_state_context:
                        logger.info(f"Avoiding misinterpreting '{abbr}' as a state abbreviation")
                        # Add a space to ensure it's not matched by our location extractor
                        message_text = message_text.replace(f" {abbr} ", f" {abbr}_ ")
            
            # Generate comprehensive abortion access response
            access_response = generate_abortion_access_response(message_text)
            
            # Update the response with the abortion access information
            if "text" in access_response:
                if "text" in response_data:
                    response_data["text"] = access_response["text"] + "\n\n" + response_data["text"]
                else:
                    response_data["text"] = access_response["text"]
            
            # Set map display if needed
            if access_response.get("show_map"):
                response_data["show_map"] = True
                
                # For travel maps, include both the origin and destination states
                if access_response.get("travel_map") or response_data.get("travel_map"):
                    response_data["map_query"] = response_data.get("map_query") or access_response.get("map_query", "")
                    response_data["travel_state"] = response_data.get("travel_state") or access_response.get("travel_state", "")
                    response_data["from_state"] = response_data.get("from_state") or access_response.get("from_state", "")
                    response_data["to_state"] = response_data.get("to_state") or access_response.get("to_state", "")
                    # Ensure we're searching for clinics in the destination state
                    destination_state = response_data.get("to_state") or access_response.get("to_state", "")
                    if destination_state:
                        # Import the policy handler's state names since app.py doesn't have access to it
                        from chatbot.policy_handler import PolicyHandler
                        policy_handler = query_processor.handlers.get("policy", None)
                        if policy_handler and hasattr(policy_handler, 'STATE_NAMES'):
                            state_name = policy_handler.STATE_NAMES.get(destination_state, destination_state)
                        else:
                            state_name = destination_state
                        response_data["zip_code"] = state_name
                        response_data["map_query"] = f"abortion clinic in {state_name}"
                        logger.info(f"Setting map to show clinics in destination state: {state_name}")
                else:
                    # Standard map query - Prioritize ZIP code if available
                    if request.zip_code:
                        response_data["zip_code"] = request.zip_code
                        response_data["map_query"] = f"abortion clinic in {request.zip_code}"
                    else:
                        response_data["zip_code"] = access_response.get("state_name", "")
                        if "map_query" in access_response:
                            response_data["map_query"] = access_response["map_query"]
            
            # Set abortion legality and state information
            if "is_legal" in access_response:
                response_data["is_legal"] = access_response["is_legal"]
            
            # Set state name 
            if "state_name" in access_response:
                state_name = access_response["state_name"]
                # Validate state name - prevent ambiguous abbreviations from being misinterpreted
                from chatbot.policy_handler import AMBIGUOUS_ABBREVIATIONS
                if state_name and state_name.lower() in AMBIGUOUS_ABBREVIATIONS:
                    logger.warning(f"Ignoring potentially misinterpreted state '{state_name}'")
                else:
                    response_data["state_name"] = state_name
            
            logger.info(f"Generated abortion access response for state: {access_response.get('state_name')}")
        
        # Then check for location-based clinic queries
        elif request.is_location_clinic_query:
            is_clinic_search = True
            logger.info("Client detected location-based clinic query")
            
            # Use client-provided ZIP code or city name if available
            zip_code = request.zip_code
            city_name = request.city_name
            
            # Set location for map display
            location_for_map = zip_code or city_name
            
            if location_for_map:
                logger.info(f"Using client-detected location: {location_for_map}")
                response_data["show_map"] = True
                response_data["zip_code"] = location_for_map
        else:
            # Fallback to server-side detection
            # Check if message contains keywords related to maps and clinics
            has_clinic_keyword = any(keyword.lower() in request.message.lower() for keyword in clinic_keywords)
            has_map_keyword = any(keyword.lower() in request.message.lower() for keyword in map_keywords)
            
            # Check if zip code is in the message (e.g., "90210")
            zip_code_match = re.search(r'\b\d{5}(?:-\d{4})?\b', request.message)
            zip_code = zip_code_match.group(0) if zip_code_match else None
            
            # Check for city name in message using regex (simple version)
            city_match = re.search(r'(?:in|near|around|at|close to)\s+([A-Za-z][A-Za-z\s]+?)(?:\s+(?:that|which|and|or|but|the|if|is|are|can|will|to)|[,.]|$)', request.message, re.IGNORECASE)
            city_name = city_match.group(1).strip() if city_match else None
            
            # Check for locations in classified data or detected_locations
            location_from_classification = None
            if "detected_locations" in response_data:
                if isinstance(response_data["detected_locations"], list) and response_data["detected_locations"]:
                    location_from_classification = response_data["detected_locations"][0]
                    
            # Also check user_location if provided
            user_zip = None
            if request.user_location and "zip_code" in request.user_location:
                user_zip = request.user_location["zip_code"]
                
            # Determine if this is a clinic search with location
            # Check for abortion keyword
            has_abortion_keyword = any(keyword.lower() in request.message.lower() for keyword in abortion_keywords)
            
            # For abortion-related queries, require BOTH clinic keyword AND location information
            if has_abortion_keyword:
                # Only treat as clinic search if both abortion keyword AND specific clinic keyword AND location are present
                if (has_clinic_keyword and (zip_code or city_name or location_from_classification or user_zip)):
                    is_clinic_search = True
                    logger.info("Detected specific clinic search with location information")
                    
                    # Set zip_code in order of priority: explicit zip in message, city name, detected location, user's zip
                    response_zip = zip_code or city_name or location_from_classification or user_zip
                    
                    # Set map display flags for clinic searches
                    response_data["show_map"] = True
                    response_data["zip_code"] = response_zip
                    
                    logger.info(f"Setting show_map=True and zip_code={response_zip} for clinic search")
                    
                    # Add a note to the response text about map display
                    if "text" in response_data and not "You can see these clinics on the map" in response_data["text"]:
                        response_data["text"] += "\n\nYou can see these clinics on the map below."
            # For non-abortion queries, only require clinic keyword and location
            elif has_clinic_keyword and (zip_code or city_name or location_from_classification or user_zip):
                is_clinic_search = True
                logger.info("Detected clinic search with location information")
                
                # Set zip_code in order of priority: explicit zip in message, city name, detected location, user's zip
                response_zip = zip_code or city_name or location_from_classification or user_zip
                
                # Set map display flags - ALWAYS set these for clinic searches
                response_data["show_map"] = True
                response_data["zip_code"] = response_zip
                
                logger.info(f"Setting show_map=True and zip_code={response_zip} for clinic search")
                
                # Add a note to the response text about map display
                if "text" in response_data and not "You can see these clinics on the map" in response_data["text"]:
                    response_data["text"] += "\n\nYou can see these clinics on the map below."
            
            # If request explicitly mentions map and has BOTH abortion AND clinic keywords, show map
            # even without specific location information
            elif has_abortion_keyword and has_clinic_keyword and has_map_keyword:
                logger.info("User explicitly requested map for abortion clinics")
                
                # Use detected location if available, then user location, then default to "your area"
                default_location = location_from_classification or user_zip or "your area"
                
                response_data["show_map"] = True
                response_data["zip_code"] = default_location
                
                logger.info(f"Setting show_map=True with default location: {default_location}")
                
                if "text" in response_data:
                    if not "You can see clinics on the map" in response_data["text"]:
                        response_data["text"] += f"\n\nYou can see abortion clinics near {default_location} on the map below."
        
        # Check if query explicitly asks for policy details
        is_policy_query = any(keyword.lower() in request.message.lower() for keyword in policy_keywords)
        
        # If policy details are requested and available, ensure they're included in the response
        if "policy_details" in response_data:
            # Get the policy details 
            policy_details = response_data["policy_details"]
            
            # Check if policy_details is not empty and actually contains data
            if policy_details and any(isinstance(v, dict) and v for v in policy_details.values()):
                # Process policy details into readable format
                policy_details_text = ""
                
                # Process gestational limits
                if "gestational_limits" in policy_details:
                    gestational = policy_details["gestational_limits"]
                    if isinstance(gestational, dict):
                        if "summary" in gestational:
                            policy_details_text += f"• {gestational['summary']}\n"
                        elif "banned_after_weeks_since_LMP" in gestational:
                            weeks = gestational["banned_after_weeks_since_LMP"]
                            if weeks == 99:
                                policy_details_text += "• Abortion is legal until viability\n"
                            else:
                                policy_details_text += f"• Abortion is prohibited after {weeks} weeks\n"
                
                # Process waiting periods
                if "waiting_periods" in policy_details:
                    waiting = policy_details["waiting_periods"]
                    if isinstance(waiting, dict):
                        if "summary" in waiting:
                            policy_details_text += f"• {waiting['summary']}\n"
                        elif "waiting_period_hours" in waiting:
                            hours = waiting["waiting_period_hours"]
                            if hours > 0:
                                policy_details_text += f"• Required waiting period: {hours} hours\n"
                            else:
                                policy_details_text += "• No mandatory waiting period\n"
                
                # Process insurance coverage
                if "insurance_coverage" in policy_details:
                    insurance = policy_details["insurance_coverage"]
                    if isinstance(insurance, dict):
                        if "summary" in insurance:
                            policy_details_text += f"• {insurance['summary']}\n"
                        elif "private_coverage_prohibited" in insurance:
                            if insurance.get("private_coverage_prohibited"):
                                policy_details_text += "• Private insurance coverage is prohibited\n"
                            else:
                                policy_details_text += "• Private insurance coverage may be available\n"
                
                # Process minors information
                if "minors" in policy_details:
                    minors = policy_details["minors"]
                    if isinstance(minors, dict):
                        if "summary" in minors:
                            policy_details_text += f"• {minors['summary']}\n"
                        elif "parental_consent_required" in minors:
                            if minors.get("parental_consent_required"):
                                policy_details_text += "• Parental consent is required for minors\n"
                            else:
                                policy_details_text += "• Parental consent is not required for minors\n"
                
                # Add policy details to the response if we have any
                if policy_details_text:
                    # Check if we're dealing with a travel state scenario
                    if "text" in response_data and "travel_state_name" in response_data:
                        state_name = response_data["travel_state_name"]
                        
                        # Replace "Policy details not available" text if present
                        if "Policy details not available" in response_data["text"]:
                            # Try multiple different replacement strategies
                            logger.info("Detected 'Policy details not available' in response, applying replacements")
                            
                            try:
                                # 1. First try the regex-based replacement
                                pattern = f"In {state_name}:([\\s\\S]*?)Traveling for abortion care"
                                replacement = f"In {state_name}:\n\n{policy_details_text}\nTraveling for abortion care"
                                
                                updated_text = re.sub(pattern, replacement, response_data["text"])
                                if updated_text != response_data["text"]:  # If a replacement was made
                                    response_data["text"] = updated_text
                                    logger.info("Replacement method 1 (regex) succeeded")
                                else:
                                    # 2. Try a simpler pattern without the "Traveling" part
                                    pattern = f"In {state_name}:(.*?)(?=\\n\\n|$)"
                                    replacement = f"In {state_name}:\n\n{policy_details_text}"
                                    updated_text = re.sub(pattern, replacement, response_data["text"], flags=re.DOTALL)
                                    
                                    if updated_text != response_data["text"]:
                                        response_data["text"] = updated_text
                                        logger.info("Replacement method 2 (simpler regex) succeeded")
                                    else:
                                        # 3. Try direct string replacement for the bullet points
                                        if "• Policy details not available" in response_data["text"]:
                                            # Replace all bullet points with policy details
                                            updated_text = response_data["text"].replace("• Policy details not available", "")
                                            
                                            # Find the insertion point after "In {state_name}:"
                                            insert_marker = f"In {state_name}:"
                                            parts = updated_text.split(insert_marker, 1)
                                            
                                            if len(parts) > 1:
                                                # Insert the policy details right after the state name
                                                updated_text = parts[0] + insert_marker + "\n\n" + policy_details_text + parts[1]
                                                response_data["text"] = updated_text
                                                logger.info("Replacement method 3 (bullet point replacement) succeeded")
                                            else:
                                                # 4. As a last resort, use text splitting
                                                before_text = response_data["text"].split("Policy details not available")[0]
                                                after_text = response_data["text"].split("Policy details not available")[-1]
                                                
                                                # Find the end of the section with multiple "Policy details not available"
                                                if "Policy details not available" in after_text:
                                                    after_parts = after_text.split("Policy details not available")
                                                    if len(after_parts) > 1:
                                                        after_text = after_parts[-1]
                                                
                                                response_data["text"] = before_text + policy_details_text + after_text
                                                logger.info("Replacement method 4 (text splitting) used as fallback")
                            except Exception as e:
                                logger.error(f"Error during policy details replacement: {str(e)}")
                                # 5. Emergency fallback - append the policy details
                                response_data["text"] += f"\n\nCorrect policy details for {state_name}:\n{policy_details_text}"
                                logger.info("Replacement method 5 (emergency append) used after error")
                        else:
                            # Use the original approach for other cases
                            insert_text = f"In {state_name}:"
                            parts = response_data["text"].split(insert_text, 1)
                            if len(parts) > 1:
                                # Add policy details after the state name
                                response_data["text"] = parts[0] + insert_text + "\n\n" + policy_details_text + "\n" + parts[1]
                            else:
                                # Add at the end as fallback
                                response_data["text"] += "\n\nAdditional policy details:\n" + policy_details_text
                    else:
                        # Add at the end as fallback
                        if "text" in response_data:
                            response_data["text"] += "\n\nAdditional policy details:\n" + policy_details_text
        
        # Add sources to the response if available
        if "citation_objects" in response_data and response_data["citation_objects"]:
            # Import urlparse here to fix the missing urlparse issue
            from urllib.parse import urlparse
            
            # Create a sources section for the response
            sources_text = "\n\nSources:"
            seen_sources = set()
            valid_sources = False
            
            for citation in response_data["citation_objects"]:
                if isinstance(citation, dict):
                    source_name = citation.get("source", "")
                    source_url = citation.get("url", "")
                    
                    # Skip invalid or undefined sources
                    if not source_name or source_name == "undefined" or "[undefined]" in source_name:
                        continue
                    
                    # Skip duplicates
                    source_key = f"{source_name}:{source_url}"
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    valid_sources = True
                    
                    # Try to extract a meaningful title from the URL
                    title = citation.get("title", source_name)
                    if source_url and title == source_name:
                        try:
                            # Parse the URL to extract a more descriptive title
                            parsed_url = urlparse(source_url)
                            path_parts = [p for p in parsed_url.path.split('/') if p]
                            
                            # Extract meaningful parts from the URL path
                            if len(path_parts) >= 3 and path_parts[-1]:
                                # Convert dashes to spaces and capitalize words
                                descriptive_title = path_parts[-1].replace('-', ' ').title()
                                if descriptive_title:
                                    title = f"{source_name}: {descriptive_title}"
                        except Exception as e:
                            logger.warning(f"Error extracting title from URL '{source_url}': {str(e)}")
                    
                    # Add the source with better formatting
                    if source_url:
                        sources_text += f"\n• {title} - {source_url}"
                    else:
                        sources_text += f"\n• {title}"
            
            # Only add sources section if we found valid sources
            if valid_sources and len(seen_sources) > 0:
                response_data["text"] += sources_text
                logger.info(f"Added {len(seen_sources)} sources to response")
        
        # Clean up response text and ensure citation_objects have proper format
        if "text" in response_data:
            # Make sure text is not empty or just a period
            if not response_data["text"] or response_data["text"].strip() == ".":
                logger.warning("Response text is empty or just a period, using fallback")
                # Check if we have policy data we can use
                if "primary_content" in response_data:
                    response_data["text"] = response_data["primary_content"]
                elif response_data.get("aspect_type") == "policy" and "state_name" in response_data:
                    state_name = response_data["state_name"]
                    response_data["text"] = f"Here's abortion policy information for {state_name}. Please check official sources like Planned Parenthood for the most up-to-date information."
                else:
                    response_data["text"] = "I apologize, but I couldn't generate a complete response to your question. Could you try rephrasing or asking another question about reproductive health?"
            
            # Remove inline citations in text
            
            # Remove numbered citations [1], [2], etc. and the actual bracket character [.
            response_data["text"] = re.sub(r'\[\d+\]', '', response_data["text"])
            response_data["text"] = re.sub(r'\[\.\s*', '', response_data["text"])
            
            # Remove citations in parentheses like (Planned Parenthood, SOURCE...)
            response_data["text"] = re.sub(r'\s?\([^)]*(?:SOURCE|source)[^)]*\)', '', response_data["text"])
            
            # Remove "SOURCE" text
            response_data["text"] = re.sub(r'\s?SOURCE.+?(?=\s|$|\.|,)', '', response_data["text"])
            
            # Remove stray brackets that might be left from citation formatting
            response_data["text"] = re.sub(r'\s?\[\.?\]', '', response_data["text"])
            
            # Remove "For more information, see sources" at the end
            response_data["text"] = re.sub(
                r"For more (?:detailed )?information,?\s*(?:you can )?(?:refer to|see|check) (?:the )?(?:resources|sources)(?:\s*from [^.]+)?\.?\s*$", 
                "", 
                response_data["text"]
            )
            # Remove citation markers at the end
            response_data["text"] = re.sub(
                r"(?:\s*\[\d+\])+\s*\.?$", 
                ".", 
                response_data["text"]
            )
            
            # Final cleanup - remove any remaining "Policy details not available" text
            if "Policy details not available" in response_data["text"]:
                logger.info("Doing final cleanup of any remaining 'Policy details not available' text")
                response_data["text"] = response_data["text"].replace("• Policy details not available", "")
                response_data["text"] = response_data["text"].replace("Policy details not available", "")
                # Clean up any empty bullet points
                response_data["text"] = re.sub(r'•\s*\n', '', response_data["text"])
                # Clean up double newlines
                response_data["text"] = re.sub(r'\n{3,}', '\n\n', response_data["text"])
        
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
        
        # Log information about citations
        logger.info(f"Response has {len(response_data.get('citations', []))} citations")
        citation_objects = response_data.get('citation_objects', [])
        
        # Just keep the citation objects from the handlers unchanged
        logger.info(f"Keeping existing citation objects: {json.dumps(citation_objects)}")
        
        # Remove duplicate citation objects
        if response_data.get("citation_objects"):
            # Track unique URLs
            unique_urls = {}
            unique_citation_objects = []
            
            for citation in response_data["citation_objects"]:
                if isinstance(citation, dict):
                    url = citation.get("url")
                    if url and url not in unique_urls:
                        unique_urls[url] = True
                        unique_citation_objects.append(citation)
                    # Keep non-URL citations
                    elif not url:
                        unique_citation_objects.append(citation)
                else:
                    # Keep string citations
                    unique_citation_objects.append(citation)
            
            # Update response with deduplicated citations
            response_data["citation_objects"] = unique_citation_objects
            # Update citations list too for consistency
            if "citations" in response_data:
                response_data["citations"] = [c.get("source") for c in unique_citation_objects if isinstance(c, dict) and "source" in c]
            
            logger.info(f"Removed duplicate citations, {len(response_data['citation_objects'])} unique citations remain")
            
        # Log the final citation objects
        logger.info(f"Citation objects: {json.dumps(response_data.get('citation_objects', []))}")
        
        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "citation_objects": response_data.get("citation_objects", [])
            }
        )
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response_data:
            response_data["graphics"] = []
            
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
        # Record response metrics
        record_time(
            metric_name='query_processing',
            elapsed_time=query_processing_time,
            session_id=session_id,
            category=response_data.get("aspect_type")) # Use .get() - sends None if key missing)
        
        # --- Record Metrics for the Response ---
        # Estimate token count and record API usage event
        response_text_length = len(response_data.get("text", ""))
        estimated_tokens = int(response_text_length / 4)
        record_api_call('chatbot_response', tokens_used=estimated_tokens, session_id=session_id)

        # Safety Score event (calls the imported or placeholder function)
        text_to_check = response_data.get("text", "") # Get text safely
        is_safe = check_response_safety(text_to_check)
        safety_score = 1.0 if is_safe else 0.0
        record_safety_score(safety_score, session_id=session_id, message_id=message_id)

        # Empathy Score event (calls the imported or placeholder function)
        empathy_score = calculate_empathy(text_to_check)
        record_empathy_score(empathy_score, session_id=session_id, message_id=message_id)

        # Memory Usage event
        record_memory_usage(session_id=session_id)
        # --- End Response Metrics ---

        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "citation_objects": response_data.get("citation_objects", [])
            }
        )

        # Track total conversation time
        total_time = time.time() - start_time
        record_time('total_response_time', total_time, session_id = session_id)
        
        # Flush metrics to ensure they're written to storage
        flush_metrics()

        # Check for clinic finder query
        if response_data.get("query_type") == "clinic_finder":
            logger.info("Detected clinic finder query in response")
            response_data["show_map"] = True
            
            # Use the exact ZIP code from the response if available
            if "exact_zip_code" in response_data:
                response_data["zip_code"] = response_data["exact_zip_code"]
                response_data["map_query"] = response_data.get("map_query", f"abortion clinic in {response_data['exact_zip_code']}")
                logger.info(f"Using exact ZIP code for map: {response_data['exact_zip_code']}")
            # Otherwise, use state name as fallback
            elif "state_name" in response_data:
                response_data["zip_code"] = response_data["state_name"]
                response_data["map_query"] = f"abortion clinic in {response_data['state_name']}"
                
            if "text" in response_data and "map" not in response_data["text"].lower():
                response_data["text"] += "\n\nYou can see available providers on the map below."

        # Add debug flag to request metadata if it's provided
        if debug and request.metadata is None:
            request.metadata = {}
        
        if debug and isinstance(request.metadata, dict):
            request.metadata["debug_mode"] = True
            logger.info("Debug mode enabled for chat request")

        # Process the user message with appropriate processor instance
        message = request.message.strip()
        
        # Ensure session ID exists
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")
        
        # Get conversation history for the session
        history = memory.get_history(session_id)
        
        # Process message with multi-aspect processor
        aspect_type = request.metadata.get("forced_aspect") if request.metadata else None
        response_data = await processor.process_query(
            message, 
            conversation_history=history, 
            user_location=request.user_location,
            session_id=session_id,
            aspect_type=aspect_type  # Pass the aspect_type parameter
        )
        
        # Debug: Log the citation_objects received from the processor
        if debug or (request.metadata and request.metadata.get("debug_mode")):
            if "citation_objects" in response_data:
                logger.info(f"DEBUG: Citation objects from processor: {len(response_data['citation_objects'])}")
                for i, citation in enumerate(response_data["citation_objects"]):
                    if isinstance(citation, dict):
                        logger.info(f"DEBUG: Citation {i}: source='{citation.get('source')}', url='{citation.get('url', 'NO URL')}'")
            else:
                logger.info("DEBUG: No citation_objects found in processor response")
        
        # Check if we need to improve citations with specific URLs
        if response_data.get("citations") and (
            not response_data.get("citation_objects") or 
            all(not isinstance(c, dict) or not c.get("url") for c in response_data.get("citation_objects", []))
        ):
            # Remove citation improvement logic
            logger.info("Using citation objects directly from handlers without modification")
            # Do not attempt to improve or modify citation objects
        
        # Add metadata fields
        response_data["session_id"] = session_id

        return response_data
        
    except Exception as e:
        # Record error
        increment_counter('errors', session_id=session_id)
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.delete("/session", status_code=200)
async def clear_session(request: SessionRequest, memory: MemoryManager = Depends(get_memory_manager)):
    """
    Clear a conversation session
    """
    try:
        result = memory.clear_session(request.session_id)
        if result:
            return {"status": "success", "message": f"Session {request.session_id} cleared"}
        else:
            return {"status": "warning", "message": f"Session {request.session_id} not found"}
            
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/history/{session_id}", status_code=200)
async def get_history(session_id: str, memory: MemoryManager = Depends(get_memory_manager)):
    """
    Get conversation history for a session
    """
    try:
        history = memory.get_history(session_id)
        return {"session_id": session_id, "messages": history}
            
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

@app.post("/feedback", status_code=200)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a message
    """
    try:
        session_id = request.session_id
        message_id = request.message_id
        rating = request.rating
        comment = request.comment

        logger.info(f"Session {session_id or 'Unknown'}: Received feedback for message {message_id}: rating={rating}")

        # Record feedback in metrics
        is_positive = rating >= 3  # Assuming rating scale is 1-5
        record_feedback(
            positive=is_positive,
            message_id=message_id,
            session_id=session_id,
            rating=rating, # Pass the actual rating
            comment=comment
        )

        # --- Removed previous commented-out code related to quality_metrics/ROUGE ---

        # In a production system, store this feedback in a database
        # For now, just log it and append to a simple JSON file
        feedback_data = {
            "message_id": message_id,
            "session_id": session_id, # Added session_id here too
            "rating": rating,
            "comment": comment,
            "timestamp": time.time() # Use time.time() for consistency, or datetime.now().isoformat()
        }

        # Append to a simple JSON file
        feedback_file = os.path.join(os.getcwd(), "user_feedback.json")
        try:
            all_feedback = [] # Initialize default
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    try:
                        content = f.read()
                        if content.strip():
                            loaded_json = json.loads(content)
                            if isinstance(loaded_json, list):
                                all_feedback = loaded_json
                            else:
                                logger.warning(f"Feedback file {feedback_file} is not a list. Overwriting.")
                        # If file is empty or whitespace, all_feedback remains []
                    except json.JSONDecodeError:
                         logger.warning(f"Could not decode JSON from {feedback_file}. Overwriting.")
                    except Exception as read_err:
                        logger.error(f"Error reading feedback file {feedback_file}: {read_err}")
                        # Decide if this is fatal. For now, proceed to overwrite/create.

            # Add new feedback
            all_feedback.append(feedback_data)

            # Write back to file
            with open(feedback_file, 'w') as f:
                json.dump(all_feedback, f, indent=2)

        except Exception as file_error:
            logger.error(f"Error saving feedback to file: {str(file_error)}")

        # Flush metrics after recording feedback
        flush_metrics()

        return {"success": True, "message": "Feedback recorded successfully"}

    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        # Use .get() for session_id as it might be Optional in the request model
        increment_counter('errors', session_id=request.session_id, metric_name='feedback_error') # Specific error counter
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

# --- END: app.py /feedback Endpoint ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for AWS load balancer
    """
    return {
        "status": "healthy",
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/debug/policy/{state_code}")
async def debug_policy_data(state_code: str):
    """
    Debug endpoint to examine raw policy data for a state
    """
    try:
        # Import the PolicyHandler
        from chatbot.policy_handler import PolicyHandler
        # Create a PolicyHandler instance
        handler = PolicyHandler()
        # Fetch policy data for the state
        policy_data = await handler._fetch_policy_data(state_code.upper())
        # Return the raw policy data
        return {"state_code": state_code.upper(), "policy_data": policy_data}
    except Exception as e:
        return {"error": str(e)}

# --- START: app.py /dashboard/metrics Endpoint ---

@app.get("/dashboard/metrics", status_code=200)
async def get_dashboard_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    session_id_filter: Optional[str] = Query(None, alias="session_id", description="Filter by Session ID")
):
    """
    Get aggregated metrics by reading and processing the raw event file (metrics.json).
    Allows filtering by date and session ID.
    """
    logger.info(f"Fetching dashboard metrics. Filters: start={start_date}, end={end_date}, session={session_id_filter}")
    try:
        metrics_file_path = "metrics.json"
        raw_events: List[Dict[str, Any]] = []

        # --- Step 1: Read the Raw Event Data File ---
        if os.path.exists(metrics_file_path):
            try:
                with open(metrics_file_path, 'r') as f:
                    content = f.read()
                    if content.strip():
                        loaded_json = json.loads(content)
                        if isinstance(loaded_json, list):
                           raw_events = loaded_json
                        else:
                           logger.warning(f"Metrics file {metrics_file_path} does not contain a list. Treating as empty.")
                    else:
                         logger.info(f"Metrics file {metrics_file_path} is empty.")
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {metrics_file_path}. Treating as empty.")
            except Exception as read_err:
                 logger.error(f"Error reading metrics file {metrics_file_path}: {read_err}")
                 # Continue with empty data for dashboard robustness

        # --- Step 2: Prepare Filters ---
        filter_start_dt: Optional[datetime] = None
        filter_end_dt: Optional[datetime] = None

        if start_date:
            try:
                filter_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                logger.warning(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD.")
        if end_date:
            try:
                filter_end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
            except ValueError:
                logger.warning(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD.")

        # --- Step 3: Filter Raw Events ---
        filtered_events: List[Dict[str, Any]] = []
        parse_errors = 0
        for event in raw_events:
            ts_str = event.get('timestamp')
            if not isinstance(ts_str, str):
                 # logger.debug(f"Skipping event with invalid timestamp: {event}") # Can be noisy
                 parse_errors += 1
                 continue

            try:
                # Handle potential timezone info ('Z' or offset) robustly
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                event_dt = datetime.fromisoformat(ts_str)

            except ValueError:
                 # Add fallback parsing if needed, but fromisoformat is preferred
                 parse_errors += 1
                 logger.warning(f"Could not parse timestamp: {ts_str}. Skipping event.")
                 continue

            # Apply Date Filters (make timezone-aware if necessary, assuming UTC for now)
            if filter_start_dt and event_dt.replace(tzinfo=None) < filter_start_dt: # Naive comparison if filter dates are naive
                continue
            if filter_end_dt and event_dt.replace(tzinfo=None) > filter_end_dt: # Naive comparison
                continue

            # Apply Session ID Filter
            if session_id_filter and event.get('session_id') != session_id_filter:
                continue

            filtered_events.append(event)

        if parse_errors > 0:
            logger.warning(f"Skipped {parse_errors} events due to timestamp parsing issues during filtering.")
        logger.info(f"Found {len(filtered_events)} events matching filters.")


        # --- Step 4: Aggregate Filtered Events ---
        total_user_messages: int = 0
        session_starts: set = set()
        response_times: List[float] = []
        query_processing_times: List[float] = []
        chatbot_api_calls: int = 0
        chatbot_tokens_used: int = 0
        feedback_positive: int = 0
        feedback_negative: int = 0
        safety_scores: List[float] = []
        empathy_scores: List[float] = []
        memory_usages: List[float] = []

        # --- Step 4.5: Aggregate Metrics by Day for Chart ---
        daily_aggregation = defaultdict(lambda: {
            'count': 0, # Count of responses evaluated for safety/empathy on this day
            'safety_scores': [],
            'empathy_scores': [],
            'response_times': [],
        })

        for event in filtered_events:
            event_type = event.get("event_type")
            metric_name = event.get("metric_name")
            value = event.get("value")
            session_id = event.get("session_id")
            ts_str = event.get('timestamp')

            # Aggregate overall stats
            if event_type == "counter":
                if isinstance(value, int):
                    if metric_name == "user_messages":
                        total_user_messages += value
                    elif metric_name == "session_starts" and session_id:
                        session_starts.add(session_id)
            elif event_type == "timer":
                if isinstance(value, (int, float)):
                    if metric_name == "total_response_time":
                        response_times.append(value)
                    elif metric_name == "query_processing":
                         query_processing_times.append(value)
            elif event_type == "api_call":
                if metric_name == "chatbot_response":
                     if isinstance(value, int): chatbot_api_calls += value
                     tokens = event.get("tokens_used", 0)
                     if isinstance(tokens, int): chatbot_tokens_used += tokens
            elif event_type == "feedback":
                 rating = event.get("rating") # Check rating for positive/negative if available
                 if rating is not None and isinstance(rating, int):
                     if rating >= 3: feedback_positive += 1 # Assuming 1-5 scale
                     else: feedback_negative += 1
                 elif isinstance(value, int): # Fallback to positive/negative flag if rating not present
                     if value == 1: feedback_positive += 1
                     elif value == 0: feedback_negative += 1
            elif event_type == "score":
                 if isinstance(value, (int, float)):
                     if metric_name == "safety_score": safety_scores.append(value)
                     elif metric_name == "empathy_score": empathy_scores.append(value)
            elif event_type == "measurement":
                 if isinstance(value, (int, float)):
                     if metric_name == "memory_usage_mb":
                         memory_usages.append(value)

            # Aggregate daily stats
            if ts_str:
                try:
                    if ts_str.endswith('Z'):
                       ts_str = ts_str[:-1] + '+00:00'
                    event_dt = datetime.fromisoformat(ts_str)
                    day_str = event_dt.strftime('%Y-%m-%d') # Group by day

                    if event_type == "score":
                        if isinstance(value, (int, float)):
                            if metric_name == "safety_score":
                                daily_aggregation[day_str]['safety_scores'].append(value)
                                daily_aggregation[day_str]['count'] += 1 # Count responses with safety score
                            elif metric_name == "empathy_score":
                                daily_aggregation[day_str]['empathy_scores'].append(value)
                                # Note: Count isn't incremented here, assuming safety is the primary driver for daily eval count
                    elif event_type == "timer":
                        if metric_name == "total_response_time" and isinstance(value, (int, float)):
                            daily_aggregation[day_str]['response_times'].append(value)

                except ValueError:
                    # Already logged during filtering pass if it failed there
                    pass
                except Exception as daily_agg_err:
                    logger.warning(f"Skipping event during daily aggregation due to error: {daily_agg_err} - Event: {event}")
                    continue

        # --- Process daily data for the chart ---
        daily_chart_data = {
            'dates': [],
            'avg_safety': [], # Corresponds to 'daily_safety' in HTML attempt (0-100%)
            'avg_empathy': [], # Can be used for 'daily_scores' in HTML if desired (0-1 score)
            'avg_response_time_ms': [] # Convert to ms for display
        }
        sorted_dates = sorted(daily_aggregation.keys())

        for day in sorted_dates:
            day_data = daily_aggregation[day]
            daily_chart_data['dates'].append(day)

            avg_safety_day = sum(day_data['safety_scores']) / len(day_data['safety_scores']) if day_data['safety_scores'] else 0
            daily_chart_data['avg_safety'].append(round(avg_safety_day * 100, 1)) # Convert to percentage 0-100

            avg_empathy_day = sum(day_data['empathy_scores']) / len(day_data['empathy_scores']) if day_data['empathy_scores'] else 0
            daily_chart_data['avg_empathy'].append(round(avg_empathy_day, 3)) # Keep as 0-1 score

            avg_resp_time_day = sum(day_data['response_times']) / len(day_data['response_times']) if day_data['response_times'] else 0
            daily_chart_data['avg_response_time_ms'].append(round(avg_resp_time_day * 1000)) # Convert s to ms

        # --- Step 5: Calculate Final Aggregated Values ---
        total_conversations = len(session_starts)
        total_responses_evaluated = len(safety_scores) # Use safety score count as proxy for evaluated responses

        avg_messages_per_conversation = (total_user_messages / total_conversations) if total_conversations > 0 else 0

        total_feedback = feedback_positive + feedback_negative
        improvement_rate = (feedback_positive / total_feedback * 100) if total_feedback > 0 else 0 # Renamed from positive_percentage

        avg_response_time_s = sum(response_times) / len(response_times) if response_times else 0
        min_response_time_s = min(response_times) if response_times else 0
        max_response_time_s = max(response_times) if response_times else 0

        avg_token_usage = (chatbot_tokens_used / chatbot_api_calls) if chatbot_api_calls > 0 else 0

        # Quality Metrics
        avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0
        avg_empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0
        # Calculate Safety Rate (e.g., % of responses with safety score >= 0.9)
        safety_threshold = 0.9 # Define your threshold for "safe"
        safe_count = sum(1 for score in safety_scores if score >= safety_threshold)
        safety_rate = (safe_count / total_responses_evaluated * 100) if total_responses_evaluated > 0 else 0

        # Placeholder for Avg Score (using avg of safety and empathy for now)
        avg_score_placeholder = ((avg_safety + avg_empathy) / 2 * 10) if (avg_safety > 0 or avg_empathy > 0) else 0 # Scale 0-10

        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        min_memory = min(memory_usages) if memory_usages else 0
        max_memory = max(memory_usages) if memory_usages else 0

        # --- Step 6: Format Output for Dashboard ---
        dashboard_data = {
            "summary": {
                "total_evaluations": total_user_messages, # Use user messages as total evaluations
                "total_conversations": total_conversations,
                "avg_messages_per_conversation": round(avg_messages_per_conversation, 1),
                "improvement_rate": round(improvement_rate, 1), # % based on positive feedback
                "safe_count": safe_count, # Needed for subtext
                "total_responses_evaluated": total_responses_evaluated, # Needed for subtext
                "positive_feedback_count": feedback_positive, # Needed for subtext
            },
            "quality_metrics": {
                 "avg_score": round(avg_score_placeholder, 1), # Placeholder average score (0-10)
                 "safety_rate": round(safety_rate, 1), # % of safe responses (0-100)
                 "avg_safety_score": round(avg_safety, 3), # Raw average safety (0-1)
                 "avg_empathy_score": round(avg_empathy, 3), # Raw average empathy (0-1)
                 # --- Relevance, Accuracy, etc. keys are removed ---
            },
            "performance_metrics": { # Group performance related items
                "response_times_ms": { # Send in ms
                    "average": round(avg_response_time_s * 1000),
                    "min": round(min_response_time_s * 1000),
                    "max": round(max_response_time_s * 1000)
                },
                "token_usage": {
                    "average": round(avg_token_usage),
                    "min": 0, # Cannot determine min tokens per call easily
                    "max": chatbot_tokens_used # Total tokens in the filtered period
                },
                "memory_usage_mb": {
                    "average": round(avg_memory, 2),
                    "min": round(min_memory, 2),
                    "max": round(max_memory, 2)
                }
            },
            "daily_metrics_trend": daily_chart_data, # Use this for the line chart
            "feedback_summary": { # Separate feedback details
                 "positive": feedback_positive,
                 "negative": feedback_negative,
                 "total": total_feedback
            },
            # Placeholder for features not yet implemented
            "rag_performance": None, # For the 'Coming Soon' section
            "top_issues": [] # Return empty list as it's not calculated
        }
        logger.info("Successfully aggregated dashboard metrics.")
        return dashboard_data

    except Exception as e:
        logger.error(f"Error processing dashboard metrics: {str(e)}", exc_info=True)
        # Return a default structure that matches the expected keys to prevent JS errors
        return {
            "summary": {"total_evaluations": 0, "total_conversations": 0, "avg_messages_per_conversation": 0.0, "improvement_rate": 0.0, "safe_count": 0, "total_responses_evaluated": 0, "positive_feedback_count": 0},
            # --- Removed avg_relevance etc. from default quality_metrics ---
            "quality_metrics": {"avg_score": 0.0, "safety_rate": 0.0, "avg_safety_score": 0.0, "avg_empathy_score": 0.0},
            "performance_metrics": {
                "response_times_ms": {"average": 0, "min": 0, "max": 0},
                "token_usage": {"average": 0, "min": 0, "max": 0},
                "memory_usage_mb": {"average": 0.0, "min": 0.0, "max": 0.0}
            },
            "daily_metrics_trend": {"dates": [], "avg_safety": [], "avg_empathy": [], "avg_response_time_ms": []},
            "feedback_summary": {"positive": 0, "negative": 0, "total": 0},
            "rag_performance": None,
            "top_issues": []
        }

# --- END: app.py /dashboard/metrics Endpoint ---

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """
    Serve the admin dashboard interface
    """
    return templates.TemplateResponse(
        "admin-dashboard.html", 
        {
            "request": request
        }
    )

@app.post("/test-multi-aspect", response_model=ChatResponse)
async def test_multi_aspect(request: ChatRequest, 
                         processor: MultiAspectQueryProcessor = Depends(get_processor),
                         memory: MemoryManager = Depends(get_memory_manager)):
    """
    Test endpoint specifically for processing multi-aspect queries
    This ensures the message is processed as a multi-aspect query even if the classifier might not detect it as such
    """
    try:
        start_time = time.time()
        logger.info(f"Received multi-aspect test request: {request.message[:50]}...")
        
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store user message in memory
        memory.add_message(
            session_id=session_id,
            message=request.message,
            role="user",
            metadata={"is_multi_aspect_test": True, **(request.metadata or {})}
        )
        
        # Get conversation history
        conversation_history = memory.get_history(session_id)
        
        # Force multi-aspect classification
        forced_classification = {
            "primary_type": "knowledge",
            "is_multi_aspect": True,
            "confidence_scores": {
                "knowledge": 0.7,
                "emotional": 0.6,
                "policy": 0.6
            },
            "topics": ["reproductive_health", "abortion", "state_laws", "emotional_support"],
            "sensitive_content": ["abortion"],
            "contains_location": True,
            "detected_locations": ["state"],
            "query_complexity": "complex"
        }
        
        # Process with forced multi-aspect
        async def process_with_forced_multi_aspect():
            # First classify the message normally
            classification = await processor.unified_classifier.classify(
                request.message, 
                conversation_history
            )
            
            # Override is_multi_aspect flag
            classification["is_multi_aspect"] = True
            
            # Boost all confidence scores to encourage multiple aspect processing
            for aspect_type in classification.get("confidence_scores", {}):
                classification["confidence_scores"][aspect_type] = max(
                    classification["confidence_scores"].get(aspect_type, 0), 
                    0.6
                )
                
            # Force query complexity to complex
            classification["query_complexity"] = "complex"
            
            # Decompose into aspects
            aspects = await processor.aspect_decomposer.decompose(
                request.message, 
                classification, 
                conversation_history
            )
            
            # Process each aspect
            aspect_responses = []
            aspect_tasks = []
            
            for aspect in aspects:
                aspect_type = aspect.get("type", "knowledge")
                
                if aspect_type in processor.handlers:
                    # Create processing task for this aspect
                    handler = processor.handlers[aspect_type]
                    task = asyncio.create_task(
                        processor._process_aspect(
                            handler=handler,
                            aspect=aspect,
                            message=request.message,
                            conversation_history=conversation_history,
                            user_location=request.user_location,
                            aspect_type=aspect_type
                        )
                    )
                    aspect_tasks.append(task)
            
            # Wait for all aspect processing to complete
            if aspect_tasks:
                aspect_responses = await asyncio.gather(*aspect_tasks)
                # Filter out None responses
                aspect_responses = [r for r in aspect_responses if r is not None]
            
            # Compose the final response
            response = await processor.response_composer.compose_response(
                message=request.message,
                aspect_responses=aspect_responses,
                classification=classification
            )
            
            return response
        
        # Process the query with forced multi-aspect
        response_data = await process_with_forced_multi_aspect()
        
        # Add message ID and session ID to response
        message_id = str(uuid.uuid4())
        response_data["message_id"] = message_id
        response_data["session_id"] = session_id
        
        if "processing_time" not in response_data:
            response_data["processing_time"] = time.time() - start_time
        
        # Add debug info
        response_data["metadata"] = {
            "is_multi_aspect_test": True,
            "aspects_count": len(response_data.get("aspect_responses", [])) if "aspect_responses" in response_data else 0,
            **(response_data.get("metadata", {}))
        }
        
        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "citation_objects": response_data.get("citation_objects", []),
                "is_multi_aspect_test": True
            }
        )
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response_data:
            response_data["graphics"] = []
            
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
            
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing multi-aspect test: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/test-knowledge")
async def test_knowledge(request: ChatRequest):
    """
    Test endpoint to directly access the knowledge handler
    """
    try:
        logger.info(f"Testing knowledge handler with query: {request.message[:100]}")
        
        # Access the knowledge handler via the processor
        knowledge_handler = query_processor.handlers.get("knowledge")
        
        if not knowledge_handler:
            return {"error": "Knowledge handler not found"}
        
        # Process the query through the knowledge handler
        response = await knowledge_handler.process_query(
            query=request.message,
            full_message=request.message
        )
        
        # Log information about citations
        logger.info(f"Knowledge response has {len(response.get('citations', []))} citations")
        logger.info(f"Knowledge citation objects: {json.dumps(response.get('citation_objects', []))}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error testing knowledge handler: {str(e)}", exc_info=True)
        return {"error": f"Error testing knowledge handler: {str(e)}"}

@app.post("/test-citations")
async def test_citations(request: ChatRequest):
    """
    Test endpoint to directly create a response with citations
    """
    try:
        # Create a test response with explicit citation objects
        response = {
            "text": "Yes, it is normal to feel both relief and sadness after an abortion. Feeling relieved not to be pregnant and yet sad at the same time can be a confusing combination, but is common and understandable. Some women will know 'in their head' that having an abortion was the right decision and do not regret it, but at the same time feel sad 'in their heart' about the end of the pregnancy. However through time and talking with others this can resolve.",
            "citations": ["Planned Parenthood", "Sexual Health Sheffield"],
            "citation_objects": [
                {
                    "source": "Sexual Health Sheffield",
                    "url": "https://www.sexualhealthsheffield.nhs.uk/wp-content/uploads/2019/06/Wellbeing-after-an-abortion.pdf",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org/learn/abortion",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time()
        }
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response:
            response["graphics"] = []
            
        logger.info(f"Test citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citations: {str(e)}"}

@app.post("/test-duplicate-citations")
async def test_duplicate_citations(request: ChatRequest):
    """
    Test endpoint that creates a response with citation URLs
    """
    try:
        # Create a test response with meaningful citation objects
        response = {
            "text": "This is a test response with multiple citations from different sources. Pregnancy occurs when sperm fertilizes an egg, which can happen during unprotected vaginal sex. A woman is most fertile during ovulation, which typically occurs around the middle of her menstrual cycle. After fertilization, the fertilized egg (zygote) travels to the uterus and implants in the uterine lining, beginning pregnancy.",
            "citations": ["Planned Parenthood", "Mayo Clinic", "WebMD"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Getting Pregnant",
                    "url": "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/how-to-get-pregnant/art-20047611",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "WebMD",
                    "title": "Understanding Early Pregnancy",
                    "url": "https://www.webmd.com/baby/understanding-conception",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citations: {str(e)}"}

@app.post("/test-inline-citations")
async def test_inline_citations(request: ChatRequest):
    """
    Test endpoint that creates a response with inline numbered citations
    """
    try:
        # Create a test response with explicit citation objects and numbered references
        response = {
            "text": "Pregnancy occurs when sperm fertilizes an egg [1], which can happen during unprotected vaginal sex. A woman is most fertile during ovulation [2], which typically occurs around the middle of her menstrual cycle. After fertilization, the fertilized egg (zygote) travels to the uterus and implants in the uterine lining, beginning pregnancy. For more detailed information, you can refer to resources from Planned Parenthood [1], Mayo Clinic [2], and WebMD [3].",
            "citations": ["Planned Parenthood", "Mayo Clinic", "WebMD"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Getting Pregnant",
                    "url": "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/how-to-get-pregnant/art-20047611",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "WebMD",
                    "title": "Understanding Early Pregnancy",
                    "url": "https://www.webmd.com/baby/understanding-conception",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test inline citation response created with {len(response['citation_objects'])} citation objects")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test inline citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test inline citations: {str(e)}"}

@app.post("/test-planned-parenthood-citations")
async def test_planned_parenthood_citations(request: ChatRequest):
    """
    Test endpoint specifically for Planned Parenthood citations that match the log example
    """
    try:
        # Create a test response that matches the logs in the user's example
        response = {
            "text": "Pregnancy occurs when sperm fertilizes an egg, which typically happens during unprotected vaginal sex. For more detailed information, you can refer to resources from Planned Parenthood [1][2][3].",
            "citations": ["Planned Parenthood", "Planned Parenthood", "Planned Parenthood"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "I Think I'm Pregnant - Now What?",
                    "url": "https://www.plannedparenthood.org/learn/teens/stds-birth-control-pregnancy/i-think-im-pregnant-now-what",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",  # Duplicate title to match the example
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test Planned Parenthood citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test Planned Parenthood citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test Planned Parenthood citations: {str(e)}"}

@app.post("/test-fixed-sources")
async def test_fixed_sources(request: ChatRequest):
    """
    Test endpoint with fixed citation objects for Planned Parenthood General
    """
    try:
        # Create a test response with fixed citation objects
        response = {
            "text": "Pregnancy occurs when sperm fertilizes an egg, which typically happens during unprotected vaginal sex. For more detailed information, you can refer to resources from Planned Parenthood General.",
            "citations": ["Planned Parenthood General"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood General",
                    "title": "Planned Parenthood General",
                    "url": "https://www.plannedparenthood.org",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test fixed source response created with citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test fixed sources: {str(e)}", exc_info=True)
        return {"error": f"Error creating test fixed sources: {str(e)}"}

@app.post("/test-multiple-sources")
async def test_multiple_sources(request: ChatRequest):
    """
    Test endpoint with multiple citation objects from the same source but with different URLs
    """
    try:
        # Create a test response with multiple citation objects from the same source
        response = {
            "text": "Birth control methods include hormonal options like pills, IUDs, and implants, as well as barrier methods like condoms. Each method has different effectiveness rates and considerations that should be discussed with a healthcare provider.",
            "citations": ["Planned Parenthood General", "Planned Parenthood General", "Planned Parenthood General"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood General",
                    "title": "Birth Control Methods",
                    "url": "https://www.plannedparenthood.org/learn/birth-control",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood General",
                    "title": "Birth Control Pills",
                    "url": "https://www.plannedparenthood.org/learn/birth-control/birth-control-pill",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood General",
                    "title": "IUD Information",
                    "url": "https://www.plannedparenthood.org/learn/birth-control/iud",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        logger.info(f"Test multiple sources response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test multiple sources: {str(e)}", exc_info=True)
        return {"error": f"Error creating test multiple sources: {str(e)}"}

@app.post("/test-emergency-contraception")
async def test_emergency_contraception(request: ChatRequest):
    """
    Test endpoint to inspect emergency contraception entries directly from the CSV file
    """
    try:
        # Check for the CSV file
        data_path = os.path.join(os.path.dirname(__file__), "data", "Planned Parenthood Data - Sahana.csv")
        
        if os.path.exists(data_path):
            # Load and inspect CSV directly
            import pandas as pd
            df = pd.read_csv(data_path)
            
            # Log the columns
            logger.info(f"CSV columns: {df.columns.tolist()}")
            
            # Look for emergency contraception in questions
            ec_rows = df[df['Question'].str.contains('emergency contraception', case=False, na=False)]
            
            # Log first 5 matching rows
            logger.info(f"Found {len(ec_rows)} rows with 'emergency contraception' in the question")
            
            # Create simplified documents
            simplified_docs = []
            for i, row in ec_rows.head(10).iterrows():
                question = row.get('Question', '')
                url = row.get('Link', '')
                
                # Create a simplified document record
                simple_doc = {
                    "index": i,
                    "question": question[:150] + "..." if len(question) > 150 else question,
                    "source": "Planned Parenthood General",
                    "url": url
                }
                simplified_docs.append(simple_doc)
                
                # Log the URL for this document
                logger.info(f"Row {i} - Question: {question[:50]}... URL: {url}")
            
            # Check if there are any documents with URLs
            docs_with_urls = [doc for doc in simplified_docs if doc.get("url")]
            logger.info(f"CSV check found {len(docs_with_urls)} documents with URLs out of {len(simplified_docs)} total documents")
            
            # Return the raw documents
            return {
                "csv_path": data_path,
                "csv_documents": simplified_docs,
                "documents_with_urls": len(docs_with_urls),
                "total_documents": len(simplified_docs)
            }
        else:
            # Return an error if the CSV file doesn't exist
            logger.error(f"CSV file not found at {data_path}")
            return {"error": f"CSV file not found at {data_path}"}
        
    except Exception as e:
        logger.error(f"Error in test emergency contraception: {str(e)}", exc_info=True)
        return {"error": f"Error testing emergency contraception: {str(e)}"}

@app.post("/test-improved-ec")
async def test_improved_ec(request: ChatRequest):
    """
    Test endpoint for emergency contraception with proper URL citations
    """
    try:
        # Check for the CSV file to get real URLs
        data_path = os.path.join(os.path.dirname(__file__), "data", "Planned Parenthood Data - Sahana.csv")
        
        # Default citation in case we can't find specific ones
        default_citations = [{
            "source": "Planned Parenthood General",
            "title": "Emergency Contraception",
            "url": "https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception",
            "accessed_date": datetime.now().strftime('%Y-%m-%d')
        }]
        
        # Try to get specific citations from the CSV
        specific_citations = []
        if os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
            
            # Look for emergency contraception in questions
            ec_rows = df[df['Question'].str.contains('emergency contraception', case=False, na=False)]
            
            # Create citations from the first few relevant rows
            for i, row in ec_rows.head(3).iterrows():
                question = row.get('Question', '')
                url = row.get('Link', '')
                
                if url and len(url) > 5:
                    citation = {
                        "source": "Planned Parenthood General",
                        "title": question[:100] + ("..." if len(question) > 100 else ""),
                        "url": url,
                        "accessed_date": datetime.now().strftime('%Y-%m-%d')
                    }
                    specific_citations.append(citation)
        
        # Use specific citations if we found them, otherwise use default
        citation_objects = specific_citations if specific_citations else default_citations
        citation_sources = ["Planned Parenthood General"] * len(citation_objects)
        
        # Create a realistic response with these citations
        response = {
            "text": "<p class='message-paragraph'>Emergency contraception is a method used to prevent pregnancy after unprotected sex, condom failure, or missed birth control pills. It works primarily by preventing or delaying ovulation.</p>\n\n<h3>Types of Emergency Contraception</h3>\n<ul class='chat-bullet-list'>\n  <li>The most common types are <strong>morning-after pills</strong> (like Plan B and ella) and the <strong>copper IUD</strong>.</li>\n  <li>Plan B One-Step and similar pills contain <strong>levonorgestrel</strong> and work best if taken within 72 hours (3 days) after unprotected sex.</li>\n  <li>ella contains <strong>ulipristal acetate</strong> and works up to 120 hours (5 days) after unprotected sex.</li>\n  <li>The <strong>copper IUD</strong> is the most effective form of emergency contraception and can be inserted up to 5 days after unprotected sex.</li>\n</ul>\n\n<h3>Effectiveness and Accessibility</h3>\n<ul class='chat-bullet-list'>\n  <li>Different types of emergency contraception have varying levels of effectiveness, with the copper IUD being the most effective.</li>\n  <li>Levonorgestrel pills like Plan B are available over-the-counter without a prescription.</li>\n  <li>ella requires a prescription from a healthcare provider.</li>\n  <li>Weight can affect how well emergency contraceptive pills work, with reduced effectiveness for people with higher body weights.</li>\n</ul>\n\n<h3>How Emergency Contraception Works</h3>\n<ul class='chat-bullet-list'>\n  <li>Emergency contraception primarily works by preventing or delaying ovulation so that no egg is released for sperm to fertilize.</li>\n  <li>It does not cause an abortion and will not terminate an existing pregnancy.</li>\n  <li>Emergency contraception should not be used as a regular form of birth control, as other methods are more effective for ongoing use.</li>\n</ul>",
            "message_id": str(uuid.uuid4()),
            "session_id": request.session_id or str(uuid.uuid4()),
            "citations": citation_sources,
            "citation_objects": citation_objects,
            "timestamp": time.time(),
            "processing_time": 0.5,
            "graphics": []
        }
        
        logger.info(f"Created test EC response with {len(citation_objects)} citation objects")
        for i, citation in enumerate(citation_objects):
            logger.info(f"Citation {i} URL: {citation.get('url', 'No URL')}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test improved EC response: {str(e)}", exc_info=True)
        return {"error": f"Error creating test improved EC response: {str(e)}"}

@app.post("/test-clean-citations")
async def test_clean_citations(request: ChatRequest):
    """
    Test endpoint with clean text and separate citations
    """
    try:
        # Create a test response with citations in a format that will be processed by the frontend
        raw_text = """Preventing sexually transmitted infections (STIs) involves a combination of strategies aimed at reducing risk during sexual activity. Here are key prevention methods:
1. **Abstinence**: The only 100% effective way to avoid STIs is to abstain from any sexual contact, including vaginal, anal, and oral sex, as well as skin-to-skin genital touching (Planned Parenthood, SOURCE).
2. **Use of Barriers**: If you choose to have sex, using condoms (external or internal) and dental dams can significantly lower your risk of STIs. These barriers help block the exchange of bodily fluids and reduce skin-to-skin contact that can transmit infections (SOURCE Mayo Clinic).
3. **Regular Testing**: Getting tested for STIs regularly is crucial, especially if you have multiple partners or engage in unprotected sex. Early detection allows for treatment, which helps maintain your health and prevents the spread of infections to others (SOURCE CDC)."""
        
        response = {
            "text": raw_text,  # This will be cleaned by the frontend
            "citations": ["Planned Parenthood", "Mayo Clinic", "CDC"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "STDs & Safer Sex",
                    "url": "https://www.plannedparenthood.org/learn/stds-hiv-safer-sex",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Mayo Clinic",
                    "title": "Sexually transmitted disease (STD) prevention",  
                    "url": "https://www.mayoclinic.org/diseases-conditions/sexually-transmitted-diseases-stds/in-depth/std-prevention/art-20044293",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "CDC",
                    "title": "How You Can Prevent Sexually Transmitted Diseases",
                    "url": "https://www.cdc.gov/std/prevention/default.htm",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log the citation objects for debugging
        logger.info("Test clean numbered citations created")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test clean citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test clean citations: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """
    Initialize components on startup
    """
    logger.info("Starting up Abby Chatbot API")
    
    # Initialize metrics collection -- added 3.22
    increment_counter('server_start')

    # In the future, we might want to initialize more components here
    # For example, loading pre-trained models, connecting to databases, etc.

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up on shutdown
    """
    logger.info("Shutting down Abby Chatbot API")
    
    # Clean up inactive sessions
    session_count = memory_manager.cleanup_inactive_sessions()
    logger.info(f"Cleaned up {session_count} inactive sessions")

@app.post("/test-duplicated-sources")
async def test_duplicated_sources(request: ChatRequest):
    """
    Test endpoint for duplicated sources with different URLs
    """
    try:
        # Create a test response with multiple citations from the same source but different URLs
        raw_text = """The signs of pregnancy can vary from person to person, and while some may experience symptoms early on, others may not notice any at all. Here are some common early signs of pregnancy:

1. **Missed Period**: This is often the first sign that prompts individuals to consider the possibility of pregnancy. [1]
2. **Swollen or Tender Breasts**: Hormonal changes can cause breast tenderness and swelling. [1]
3. **Nausea and/or Vomiting**: Commonly referred to as "morning sickness," this can occur at any time of the day. [2]
4. **Fatigue**: Many people feel unusually tired during early pregnancy due to hormonal changes. [1]
5. **Bloating**: Some may experience bloating similar to what is felt during PMS. [3]
6. **Constipation**: Hormonal changes can slow down the digestive system. [2]
7. **Frequent Urination**: Increased urination can occur as the uterus expands and puts pressure on the bladder. [3]

It's important to note that these symptoms can also be caused by other factors, such as stress or hormonal fluctuations unrelated to pregnancy. Therefore, the only definitive way to confirm a pregnancy is by taking a pregnancy test, which can be done at home or at a healthcare provider's office. [1]

If you suspect you might be pregnant, consider taking a pregnancy test after your missed period for the most accurate results. If the test is positive, it's advisable to schedule an appointment with a healthcare provider to discuss your options and ensure your health. [2]"""
        
        # Create citation objects with specific URLs
        citations = [
            {
                "source": "Planned Parenthood",
                "title": "Pregnancy Symptoms",
                "url": "https://www.plannedparenthood.org/learn/pregnancy/pregnancy-symptoms",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            },
            {
                "source": "Planned Parenthood",
                "title": "Morning Sickness & Nausea During Pregnancy",
                "url": "https://www.plannedparenthood.org/learn/pregnancy/morning-sickness",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            },
            {
                "source": "Planned Parenthood",
                "title": "Pregnancy Tests & Other Services",
                "url": "https://www.plannedparenthood.org/learn/pregnancy/pregnancy-tests",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        # Log each citation and its URL for debugging
        for i, citation in enumerate(citations):
            logger.info(f"Citation {i+1}: {citation['title']} - {citation['url']}")
        
        response = {
            "text": raw_text,
            "citations": ["Planned Parenthood", "Planned Parenthood", "Planned Parenthood"],
            "citation_objects": citations,
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log the citation objects for debugging
        logger.info("Test duplicated sources created")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test duplicated sources: {str(e)}", exc_info=True)
        return {"error": f"Error creating test duplicated sources: {str(e)}"}

@app.post("/test-citation-links")
async def test_citation_links(request: ChatRequest):
    """
    Test endpoint specifically for citation links in text
    """
    try:
        # Create a test response with clear numbered citations and distinct URLs
        raw_text = """Pregnancy occurs when a sperm cell fertilizes an egg, leading to the implantation of the fertilized egg in the lining of the uterus. Here's a step-by-step overview of how this process happens:

1. **Ovulation**: About halfway through a woman's menstrual cycle, a mature egg is released from the ovary in a process called ovulation. The egg then travels through the fallopian tube towards the uterus. [1]

2. **Fertilization**: If sperm are present in the vagina (usually from vaginal intercourse), they can swim through the cervix and into the uterus, eventually reaching the fallopian tubes. If a sperm cell meets the egg within about 12-24 hours after ovulation, fertilization occurs. It only takes one sperm to fertilize the egg. [2]

3. **Development of the Fertilized Egg**: After fertilization, the fertilized egg (now called a zygote) begins to divide and grow as it moves down the fallopian tube toward the uterus. This process takes about 3-4 days. [3]

4. **Implantation**: Once the fertilized egg reaches the uterus, it floats for a couple of days before implanting itself into the thick, spongy lining of the uterus. This implantation usually occurs about 6-10 days after fertilization and marks the official start of pregnancy. [2]

5. **Hormonal Changes**: After implantation, the developing embryo releases hormones that prevent the uterine lining from shedding, which is why menstruation does not occur during pregnancy. [1]

If the egg is not fertilized or if the fertilized egg does not implant successfully, the body will shed the uterine lining during menstruation."""
        
        response = {
            "text": raw_text,
            "citations": ["Planned Parenthood", "Planned Parenthood", "Planned Parenthood"],
            "citation_objects": [
                {
                    "source": "Planned Parenthood",
                    "title": "How Pregnancy Happens",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/how-pregnancy-happens",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "What happens during fertilization?",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/fertility/what-happens-fertilization",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "title": "Pregnancy Tests & Care",
                    "url": "https://www.plannedparenthood.org/learn/pregnancy/pregnancy-tests",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time(),
            "graphics": []
        }
        
        # Log detailed information about each citation for debugging
        logger.info("=== TEST CITATION LINKS RESPONSE ===")
        logger.info(f"Text has {len(response['citation_objects'])} citation objects")
        
        for i, citation in enumerate(response["citation_objects"]):
            logger.info(f"Citation {i+1}:")
            logger.info(f"  - Source: {citation['source']}")
            logger.info(f"  - Title: {citation['title']}")
            logger.info(f"  - URL: {citation['url']}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citation links: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citation links: {str(e)}"}

@app.get("/admin/dashboard")
async def admin_dashboard(request: Request, token: str = None):
    """
    Admin dashboard to monitor chatbot usage and performance
    """
    # Simple authentication - in production you would use proper auth
    if not token or token != os.environ.get("ADMIN_TOKEN", "admin_secret"):
        return templates.TemplateResponse(
            "login.html", 
            {"request": request}
        )
    
    # Get some basic stats for the dashboard
    try:
        # Count total conversations
        total_conversations = 0
        if os.path.exists("data/sessions"):
            total_conversations = len(os.listdir("data/sessions"))
        
        # Count total messages
        total_messages = 0
        if os.path.exists("logs/messages.jsonl"):
            with open("logs/messages.jsonl", "r") as f:
                total_messages = sum(1 for _ in f)
        
        # Get most common queries (simplified)
        common_queries = [
            {"query": "Abortion options", "count": 57},
            {"query": "Birth control methods", "count": 42},
            {"query": "Pregnancy symptoms", "count": 38},
            {"query": "STI prevention", "count": 31},
            {"query": "Mental health resources", "count": 24}
        ]
        
        return templates.TemplateResponse(
            "admin-dashboard.html", 
            {
                "request": request,
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "common_queries": common_queries,
                "api_usage": {"tokens": 24500, "cost": "$0.49"}
            }
        )
    except Exception as e:
        logger.error(f"Error loading admin dashboard: {e}")
        return HTMLResponse(f"<h1>Error loading dashboard</h1><p>{str(e)}</p>")

@app.get("/favicon.ico")
async def favicon():
    """
    Serve the favicon (using logo.png since favicon.ico doesn't exist)
    """
    return FileResponse("static/images/logo.png")

@app.post("/test-policy")
async def test_policy(request: ChatRequest):
    """
    Test endpoint to directly access the policy handler
    """
    try:
        logger.info(f"Testing policy handler with query: {request.message[:100]}")
        
        # Access the policy handler via the processor
        policy_handler = query_processor.handlers.get("policy")
        
        if not policy_handler:
            return {"error": "Policy handler not found"}
        
        # Process the query through the policy handler
        response = await policy_handler.process_query(
            query=request.message,
            full_message=request.message
        )
        
        # Log information about response
        logger.info(f"Policy response type: {type(response)}")
        logger.info(f"Policy response keys: {list(response.keys() if isinstance(response, dict) else [])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error testing policy handler: {str(e)}", exc_info=True)
        return {"error": f"Error testing policy handler: {str(e)}"}

# Add a test endpoint for clinic data
@app.get("/api/clinics/{location}")
async def get_clinic_data(location: str, request: Request):
    """
    Get clinic data for a specific location - for testing map functionality
    """
    logger.info(f"Clinic data requested for location: {location}")
    
    # Sample clinic data based on location
    clinics = []
    
    # Detroit clinics
    if "detroit" in location.lower():
        clinics = [
            {
                "name": "Planned Parenthood - Detroit Health Center",
                "address": "4229 Cass Ave, Detroit, MI 48201",
                "distance": 1.5,
                "phone": "(313) 833-7080",
                "website": "https://www.plannedparenthood.org/health-center/michigan/detroit/48201/detroit-health-center-2651-90430"
            },
            {
                "name": "Scotsdale Women's Center",
                "address": "26098 Woodward Ave, Royal Oak, MI 48067",
                "distance": 10.2,
                "phone": "(248) 559-9440",
                "website": "https://scotsdalewomens.com/"
            },
            {
                "name": "Northland Family Planning Center",
                "address": "35000 Ford Rd, Westland, MI 48185",
                "distance": 18.7,
                "phone": "(734) 722-6676",
                "website": "https://northlandfamily.com/"
            }
        ]
    # Default clinics
    else:
        clinics = [
            {
                "name": "Planned Parenthood Health Center",
                "address": f"123 Main St, {location}, USA",
                "distance": 3.5,
                "phone": "(800) 230-PLAN"
            },
            {
                "name": "Family Planning Associates",
                "address": f"456 Health Ave, {location}, USA",
                "distance": 5.1,
                "phone": "(800) 435-8742"
            }
        ]
    
    return {"location": location, "clinics": clinics}

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

@app.route('/map-test')
def map_test():
    """Route to test Google Maps API functionality"""
    return render_template('map_test.html')

@app.post("/test-knowledge-debug")
async def test_knowledge_debug(request: ChatRequest, processor: MultiAspectQueryProcessor = Depends(get_processor)):
    """
    Test endpoint that directly uses the knowledge handler for debugging URLs
    """
    try:
        from chatbot.knowledge_handler import KnowledgeHandler
        
        # Create a test query
        test_query = "How long do I have to take emergency contraception?"
        
        # Create a knowledge handler instance
        api_key = os.getenv("OPENAI_API_KEY")
        knowledge_handler = KnowledgeHandler(api_key=api_key)
        
        # Process query directly with the knowledge handler
        knowledge_response = await knowledge_handler.process_query(
            query=test_query,
            full_message=test_query,
            conversation_history=[]
        )
        
        # Examine citation objects in the response
        logger.info(f"DEBUG: Direct knowledge handler citation objects: {len(knowledge_response.get('citation_objects', []))} found")
        for i, citation in enumerate(knowledge_response.get('citation_objects', [])):
            if isinstance(citation, dict):
                logger.info(f"DEBUG: Citation {i}: source='{citation.get('source')}', url='{citation.get('url', 'NO URL')}'")
        
        # Process with regular processor for comparison
        processor_response = await processor.process_query(
            message=test_query,
            conversation_history=[],
            aspect_type="knowledge"
        )
        
        # Compare citation objects from processor
        logger.info(f"DEBUG: Processor citation objects: {len(processor_response.get('citation_objects', []))} found")
        for i, citation in enumerate(processor_response.get('citation_objects', [])):
            if isinstance(citation, dict):
                logger.info(f"DEBUG: Processor Citation {i}: source='{citation.get('source')}', url='{citation.get('url', 'NO URL')}'")
        
        # Create a combined debug response
        return {
            "knowledge_handler_response": knowledge_response,
            "processor_response": processor_response,
            "debug_message": "Check server logs for detailed citation information"
        }
        
    except Exception as e:
        logger.error(f"Error in test knowledge debug: {str(e)}", exc_info=True)
        return {"error": f"Error in test knowledge debug: {str(e)}"}