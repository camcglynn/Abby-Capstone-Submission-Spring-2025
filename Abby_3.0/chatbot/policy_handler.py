# --- START OF FILE policy_handler.py ---

import logging
import os
import asyncio
import aiohttp
import json
import time
import re
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime

# Import for OpenAI
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Import for Google Maps
try:
    import googlemaps
    from geopy.geocoders import Nominatim
except ImportError:
    googlemaps = None
    Nominatim = None

# Import for ZIP code lookup
try:
    from pyzipcode import ZipCodeDatabase
except ImportError:
    ZipCodeDatabase = None

# Import abortion access utilities
try:
    # Try direct import first
    from utils.abortion_access import get_less_restrictive_states, ABORTION_LEGAL_STATES, is_abortion_legal_in_state
except ImportError:
    try:
        # Try relative import
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        from utils.abortion_access import get_less_restrictive_states, ABORTION_LEGAL_STATES, is_abortion_legal_in_state
    except ImportError:
        # Fallback implementations if utils module is not available
        logger.warning("Unable to import abortion_access module. Using fallback definitions.")
        ABORTION_LEGAL_STATES = []
        def get_less_restrictive_states(state_code: str) -> List[str]:
            return []
        def is_abortion_legal_in_state(state_code: str) -> bool:
            return False

logger = logging.getLogger(__name__)

# Define ambiguous state abbreviations that should not be detected in lowercase
AMBIGUOUS_ABBREVIATIONS = {"in", "on", "at", "me", "hi", "ok", "or", "la", "pa", "no", "so", "de", "oh", "co", "wa", "va"} # Expanded list

def is_ambiguous_state_code(state_code: str) -> bool:
    """
    Check if a state code is ambiguous and could be mistaken for a common English word.
    
    Args:
        state_code (str): Two-letter state code to check
        
    Returns:
        bool: True if the state code is considered ambiguous
    """
    ambiguous_codes = {
        "in", "or", "me", "hi", "oh", "ok", "pa", "la", "co", "wa", "va", "de", 
        "ar", "ca", "id", "ma", "mo", "ut"
    }
    return state_code.lower() in ambiguous_codes

class PolicyHandler:
    """
    Handles policy-related queries with location context awareness
    """

    def __init__(self, api_key: Optional[str] = None, policy_api_base_url: Optional[str] = None):
        """
        Initialize the policy handler

        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            policy_api_base_url (Optional[str]): Base URL for the abortion policy API
        """
        logger.info("Initializing PolicyHandler")

        # Load environment variables or use provided values
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.abortion_policy_api_key = os.environ.get("ABORTION_POLICY_API_KEY", "")  # Default empty string
        self.policy_api_base_url = policy_api_base_url or os.environ.get("POLICY_API_BASE_URL", "https://api.abortionpolicyapi.com/v1/")
        self.openai_client = None

        if self.openai_api_key and AsyncOpenAI:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            # Set the default OpenAI model
            self.openai_model = "gpt-4o" # Changed default model
            logger.info("OpenAI client initialized for policy handler")
        else:
            logger.warning("OpenAI client not available, using template-based fallback")

        # Initialize Google Maps client if API key is available
        self.gmaps = None
        self.nominatim = None
        google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
        if google_maps_api_key and googlemaps:
            if not google_maps_api_key.strip():
                logger.warning("Google Maps API key is empty. Map functionality will be disabled.")
            else:
                try:
                    # Initialize the client
                    self.gmaps = googlemaps.Client(key=google_maps_api_key)
                    
                    # Test API key with a simple geocode request to validate it
                    try:
                        # Test with a simple geocode request using a well-known location
                        test_result = self.gmaps.geocode("Washington DC, USA")
                        if test_result and len(test_result) > 0:
                            logger.info("Google Maps API key validated successfully")
                        else:
                            logger.warning("Google Maps API geocode test returned no results. API key may be restricted.")
                    except Exception as test_e:
                        error_msg = str(test_e)
                        if "REQUEST_DENIED" in error_msg or "API key" in error_msg or "not authorized" in error_msg:
                            logger.error(f"Google Maps API key validation failed: {error_msg}. Map functionality will be disabled.")
                            self.gmaps = None  # Disable the client
                        else:
                            logger.warning(f"Google Maps API test had a non-critical error: {error_msg}. Will still try to use it.")
                    
                    # Initialize Nominatim as a backup for geocoding
                    if Nominatim:
                        self.nominatim = Nominatim(user_agent="abby_abortion_chatbot")
                    logger.info("Map clients initialization complete")
                except Exception as e:
                    logger.error(f"Failed to initialize Google Maps client: {str(e)}")
        else:
            if not google_maps_api_key:
                logger.warning("Google Maps API key is not set in environment variables. Map functionality will be disabled.")
            elif not googlemaps:
                logger.warning("Google Maps Python client is not installed. Map functionality will be disabled.")

        # Fallback clinic data when API calls fail
        self.fallback_clinics = []

        # States mapping - include full state names for better matching
        self.STATE_NAMES = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
            "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
            "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
            "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
            "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
            "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
            "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
            "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
            "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
            "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
            "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
            "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
        }

        # Create lowercase state names for searches
        self.STATE_NAMES_LOWER = {k.lower(): k for k in self.STATE_NAMES.keys()}
        for k, v in self.STATE_NAMES.items():
            self.STATE_NAMES_LOWER[v.lower()] = k

        # Initialize cache for policy API responses
        self.policy_cache = {}
        self.cache_ttl = 86400  # 24 hours - policy data doesn't change often

        # Keep track of state associations for sessions
        self.session_state_cache = {}  # Maps session_id -> state_code

        # Endpoints for the Abortion Policy API
        self.api_base_url = self.policy_api_base_url
        self.api_endpoints = {
            "waiting_periods": "waiting_periods",
            "insurance_coverage": "insurance_coverage",
            "gestational_limits": "gestational_limits",
            "minors": "minors"
            # Added 'general_information' or similar if available
            # "general_information": "general_information"
        }

    def _create_empty_response(self, start_time):
        """Creates a default empty response object."""
        return {
            "text": "",
            "aspect_type": "policy",
            "confidence": 0.0,
            "citations": [],
            "citation_objects": [],
            "processing_time": time.time() - start_time,
            "primary_content": "",
            "question_answered": False,
            "needs_state_info": False # Usually false for empty/greeting
        }

    async def process_query(self, query: str, full_message: Optional[str] = None, 
                        original_message: Optional[str] = None, user_location: Optional[Dict] = None,
                        conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a policy query and return a response.

        Args:
            query (str): The processed query (may be from classifier/decomposer)
            full_message (Optional[str]): Full user message
            original_message (Optional[str]): Original unprocessed user message
            user_location (Optional[Dict]): User location data if available
            conversation_history (Optional[List[Dict]]): Conversation history

        Returns:
            Dict[str, Any]: Response with policy information or location request
        """
        start_time = time.time()
        
        try:
            # Use the most complete version of the user's input
            actual_user_input = original_message or full_message or query
            logger.info(f"Processing policy query: {query[:100]}... | Original input: {actual_user_input[:100]}...")

            # --- NEW: Check for multi-state comparison request ---
            # Check for multiple states mentioned in the current query
            all_mentioned_states = self._get_all_state_mentions(query, actual_user_input)

            if len(all_mentioned_states) > 1:
                logger.info(f"Detected multi-state comparison request for: {all_mentioned_states}")
                # Call the comparison handler directly and return its result
                comparison_response = await self._handle_state_comparison(query, all_mentioned_states, actual_user_input)
                # Add processing time and other metadata if not already done by _handle_state_comparison
                comparison_response["processing_time"] = time.time() - start_time
                comparison_response.setdefault("aspect_type", "policy")
                comparison_response.setdefault("citations", [])
                comparison_response.setdefault("citation_objects", [])
                # Ensure required keys exist if _handle_state_comparison doesn't guarantee them
                return comparison_response
            
            # Continue with single state processing if not a comparison request
            # --- END NEW: Check for multi-state comparison ---

            # --- Check cached responses ---
            
            # First check if we already answered for a state recently
            # Use states found in the query to check cache
            cached_states = set()
            for state_code in self.STATE_NAMES:
                if state_code in self.policy_cache and re.search(r'\b' + re.escape(state_code) + r'\b', query, re.IGNORECASE):
                    cached_states.add(state_code)
                elif self.STATE_NAMES[state_code] and re.search(r'\b' + re.escape(self.STATE_NAMES[state_code]) + r'\b', query, re.IGNORECASE):
                    cached_states.add(state_code)
            
            if cached_states:
                # If we find multiple states in query, we don't know which to respond with
                if len(cached_states) > 1:
                    logger.info(f"Found multiple states ({', '.join(cached_states)}) in query, not using cache")
                else:
                    # Use the cache for the found state
                    state_code = list(cached_states)[0]
                    cached_response = self.policy_cache.get(state_code)
                    if cached_response and time.time() - cached_response.get("timestamp", 0) < self.cache_ttl: # Use self.cache_ttl instead of CACHE_TTL
                        logger.info(f"Using cached response for {state_code}")
                        # Clone the cache so we don't modify the cached version
                        response_structure = cached_response.copy()
                        # Update the timestamp and processing time
                        response_structure["timestamp"] = time.time()
                        response_structure["processing_time"] = time.time() - start_time
                        return response_structure

            # --- Policy query handling logic ---
            
            # --- UPDATED: Strict prioritization in the following order:
            # 1. ZIP code in current query (highest priority)
            # 2. Explicit state mention in current query
            # 3. User location data
            # 4. Conversation history (lowest priority)
            # ---

            state_code = None
            zip_state_source = None
            
            # 1. HIGHEST PRIORITY: Check for ZIP code in current query
            zip_code = self._get_zip_code_from_query(actual_user_input)
            if zip_code:
                state_from_zip = self._get_state_from_zip(zip_code)
                if state_from_zip:
                    state_code = state_from_zip
                    zip_state_source = zip_code
                    logger.info(f"Prioritizing state {state_code} from ZIP code {zip_code}")
                else:
                    logger.warning(f"Found ZIP '{zip_code}' but couldn't map to a state")
            
            # 2. Check for explicit state mention in current query ONLY if no ZIP code state found
            if not state_code:
                state_from_query = self._get_state_from_query(query)
                if state_from_query:
                    state_code = state_from_query
                    logger.info(f"Using explicit state mention from current query: {state_code}")
            
            # 3. If still no state, check user location data
            if not state_code and user_location:
                # Try to get state directly from user_location
                loc_state_code = user_location.get('state_code', '').upper()
                if loc_state_code and len(loc_state_code) == 2 and loc_state_code in self.STATE_NAMES:
                    state_code = loc_state_code
                    logger.info(f"Using state from user location data: {state_code}")
                
                # If no state code directly, try state name
                elif not state_code and 'state' in user_location:
                    state_name = user_location.get('state', '')
                    if state_name:
                        code_from_name = self._get_state_from_query(state_name)
                        if code_from_name:
                            state_code = code_from_name
                            logger.info(f"Using state from user location state name: {state_code}")
                
                # If still no state, try ZIP from user location
                if not state_code:
                    loc_zip = user_location.get('zip', '') or user_location.get('zip_code', '') or user_location.get('postal_code', '')
                    if loc_zip:
                        state_from_zip = self._get_state_from_zip(loc_zip)
                        if state_from_zip:
                            state_code = state_from_zip
                            zip_state_source = loc_zip
                            logger.info(f"Using state from user location ZIP: {state_code} (from {loc_zip})")
            
            # 4. If still no state, check conversation history (lowest priority)
            if not state_code and conversation_history:
                state_from_history = self._get_state_from_history(None, conversation_history)
                if state_from_history:
                    state_code = state_from_history
                    logger.info(f"Using state from conversation history: {state_code}")
                    
            # Now that we have a state_code (or None), ensure we have the ZIP code if available
            # Important: This is different from the ZIP used to determine state above
            travel_zip_code = zip_code  # Save the original query ZIP for use in potential travel context
            if not zip_code and user_location:
                zip_code = user_location.get('zip', '') or user_location.get('zip_code', '') or user_location.get('postal_code', '')

            # If we have a state code, generate a policy response
            if state_code:
                logger.info(f"Generating policy response for state: {state_code}")
                
                # Get abortion access info (from API or fallback data)
                policy_data = await self._fetch_policy_data(state_code)
                
                # --- NEW: Check if this is a restrictive state and find the nearest less restrictive state ---
                is_primary_state_restrictive = not self._is_abortion_legal(state_code, policy_data)
                logger.info(f"Primary state {state_code} is restrictive: {is_primary_state_restrictive}")

                travel_state_code = None
                travel_state_name = None
                travel_policy_data = None
                requires_travel = False

                if is_primary_state_restrictive:
                    requires_travel = True
                    nearby_less_restrictive_codes = self._get_less_restrictive_states(state_code)
                    
                    if nearby_less_restrictive_codes:
                        travel_state_code = nearby_less_restrictive_codes[0]  # Select the first/closest one
                        travel_state_name = self.STATE_NAMES.get(travel_state_code, travel_state_code)
                        logger.info(f"Primary state {state_code} is restrictive. Suggesting travel to nearby state: {travel_state_code}")
                        # Fetch policy data for the travel state
                        travel_policy_data = await self._fetch_policy_data(travel_state_code)
                    else:
                        logger.warning(f"State {state_code} is restrictive, but no nearby less restrictive states found by utility.")
                        # Decide how to handle this - maybe just show restrictive state info
                        requires_travel = False  # Reset if no viable travel state found
                # --- END NEW: Restrictive state handling ---
                
                # Determine citations based on API success
                policy_api_succeeded = not policy_data.get("error", True) # API succeeded if error is False or not present

                if policy_api_succeeded:
                    # Use specific citations since we got data from the API
                    citation_objects = self._get_policy_citations(state_code)
                    logger.info(f"Using specific policy citations for {state_code} (API success)")
                else:
                    # API failed or returned fallback - use only general resources
                    # Only AbortionFinder citation for fallback
                    citation_objects = [
                        {
                            "id": "abortionfinder",
                            "source": "AbortionFinder",
                            "title": "Find Verified Abortion Care",
                            "url": "https://www.abortionfinder.org/",
                            "accessed_date": datetime.now().strftime('%Y-%m-%d')
                        }
                    ]
                    logger.info(f"Using only AbortionFinder citation for {state_code} (API fallback/error)")

                # --- NEW: Add travel state citations if needed ---
                if requires_travel and travel_state_code and travel_policy_data:
                    travel_citations = self._get_policy_citations(travel_state_code)
                    # Add travel state citations if not already present
                    for citation in travel_citations:
                        if not any(c["id"] == citation["id"] for c in citation_objects):
                            citation_objects.append(citation)
                    logger.info(f"Added citations for travel state {travel_state_code}")
                # --- END NEW: Travel citations ---

                # Use the contextual query for better OpenAI generation
                contextual_query = full_message or query
                logger.info(f"Using contextual query for generation: {contextual_query[:100]}...")

                # Generate the response using OpenAI if available
                if self.openai_client and policy_api_succeeded: # Pass policy_data only if successful
                    try:
                        # Generate response with contextual query, including travel state info if needed
                        response_text = await self._generate_with_openai(
                            contextual_query, 
                            state_code, 
                            policy_data,
                            travel_state_code,
                            travel_policy_data
                        )
                        openai_failed = False
                    except Exception as e:
                        logger.error(f"OpenAI error for policy response: {str(e)}")
                        response_text = self._format_policy_data(
                            state_code, 
                            policy_data,
                            travel_state_code,
                            travel_policy_data
                        )
                        openai_failed = True
                else:
                    # Fallback to template-based formatting
                    logger.info("Using template-based policy response.")
                    response_text = self._format_policy_data(
                        state_code, 
                        policy_data,
                        travel_state_code,
                        travel_policy_data
                    )
                    openai_failed = True
                
                # If zip_code not set already, try to extract from query or user location
                if not zip_code:
                    zip_code = self._get_zip_code_from_query(query)
                    if not zip_code and user_location and 'zip_code' in user_location:
                        zip_code = user_location['zip_code']
                
                # Create structured response
                response = {
                    "text": response_text,
                    "aspect_type": "policy",
                    "primary_content": response_text, # Ensure primary content is set
                    "state_code": state_code,
                    "state_name": self.STATE_NAMES.get(state_code, state_code),
                    "question_answered": policy_api_succeeded, # Answered only if API worked
                    "needs_state_info": False,
                    "citations": [c['source'] for c in citation_objects],
                    "citation_objects": citation_objects,
                    "processing_time": time.time() - start_time,
                    "timestamp": time.time(),
                    "policy_details": self._extract_structured_details(policy_data, state_code) if policy_api_succeeded else {}, # Only include details if API worked
                    "is_legal": self._is_abortion_legal(state_code, policy_data), # Legality check can use fallback data
                    # --- NEW: Add travel-related fields ---
                    "requires_travel": requires_travel,
                    "travel_state_code": travel_state_code,
                    "travel_state_name": travel_state_name,
                    "travel_policy_details": self._extract_structured_details(travel_policy_data, travel_state_code) if travel_policy_data else None,
                    # --- END NEW: Travel fields ---
                }

                 # --- Handle ZIP code and map info if relevant ---
                if zip_code:
                     # --- NEW: Decide which state to use for clinic search ---
                     zip_to_search = zip_code  # Default to original zip
                     state_for_search = state_code  # Default to original state

                     if requires_travel and travel_state_code:
                         logger.info(f"Travel required. Searching for clinics relative to travel state {travel_state_code}.")
                         # If original ZIP is likely far from travel state, log this
                         if zip_code:
                              logger.warning(f"Using original ZIP {zip_code} for clinic search, but recommended travel state is {travel_state_code}. Map results might be centered far from optimal clinics.")
                         state_for_search = travel_state_code  # Map should center/relate to the travel state
                     # --- END NEW: Clinic search state determination ---
                     
                     response["zip_code"] = zip_to_search
                     # Check if gmaps client is available before finding clinics
                     if self.gmaps:
                         map_data = self._find_nearby_clinics(zip_to_search, radius_miles=50, state_context=state_for_search)
                         
                         # Check if we got a valid map_data with clinics or an API error
                         if "api_error" in map_data:
                             error_message = map_data["api_error"]
                             logger.warning(f"Google Maps API error: {error_message}")
                             
                             # Don't show the map if there's an API error
                             response["show_map"] = False
                             response["map_data"] = {"clinics": [], "error_message": error_message}
                             
                             # Add note about the API error to the response text if it's an authorization issue
                             if "not authorized" in error_message or "API key" in error_message:
                                 api_error_note = "\n\n<p class='message-paragraph'>I'm unable to show nearby clinics due to an API configuration issue. Please contact support for assistance.</p>"
                                 response["text"] += api_error_note
                                 response["primary_content"] = response["text"]
                         else:
                             # We have valid map data with clinics
                             response["map_data"] = map_data
                             # Correctly set show_map based on whether the *returned* clinics list is non-empty
                             response["show_map"] = bool(map_data.get("clinics")) # Use .get() for safety
                             
                             # Use actual clinic count in text
                             actual_clinic_count = len(map_data.get("clinics", []))
                             if actual_clinic_count > 0:
                                 # --- NEW: Adjust messaging based on primary vs travel state ---
                                 if requires_travel and travel_state_code and state_for_search == travel_state_code:
                                     plural_s = "s" if actual_clinic_count > 1 else ""
                                     clinic_count_text = f"\n\n<p class='message-paragraph'>I found {actual_clinic_count} clinic{plural_s} in {travel_state_name} near {zip_to_search} that may offer relevant services. Please check the map for details.</p>"
                                 else:
                                     plural_s = "s" if actual_clinic_count > 1 else ""
                                     clinic_count_text = f"\n\n<p class='message-paragraph'>I found {actual_clinic_count} clinic{plural_s} near {zip_to_search} that may offer relevant services. Please check the map for details.</p>"
                                 # --- END NEW: Adjusted messaging ---
                                 
                                 # Append to both text and primary_content
                                 response["text"] += clinic_count_text
                                 response["primary_content"] = response["text"]
                             elif not response.get("api_error"): # Only add 'no clinics found' if there wasn't an API error
                                 # --- NEW: Adjust messaging based on primary vs travel state ---
                                 if requires_travel and travel_state_code and state_for_search == travel_state_code:
                                     no_clinic_text = f"\n\n<p class='message-paragraph'>I searched for clinics in {travel_state_name} near {zip_to_search} but didn't find relevant results matching our criteria in that specific area. You may want to try searching a wider area or use resources like AbortionFinder.org.</p>"
                                 else:
                                     no_clinic_text = f"\n\n<p class='message-paragraph'>I searched for clinics near {zip_to_search} but didn't find relevant results matching our criteria in that specific area. You may want to try searching a wider area or use resources like AbortionFinder.org.</p>"
                                 # --- END NEW: Adjusted messaging ---
                                 
                                 response["text"] += no_clinic_text
                                 response["primary_content"] = response["text"]
                     else:
                         logger.warning(f"Google Maps not available, cannot search for clinics near {zip_to_search}")
                         response["show_map"] = False

                # Ensure show_map exists even if no zip code or gmaps
                if "show_map" not in response:
                     response["show_map"] = False

                # Cache the successful response
                if not policy_data.get("error"):
                    # Store a copy to prevent modification issues
                    cache_data = response.copy()
                    # Don't cache large map data
                    cache_data.pop("map_data", None)
                    self.policy_cache[state_code] = cache_data

                return response

            # If no state code is identified, request location info
            else:
                logger.info("No state context found, requesting location information.")
                # Use only AbortionFinder citation for location request
                default_citations = [
                    {
                        "id": "abortionfinder", 
                        "source": "AbortionFinder",
                        "title": "Find Verified Abortion Care", 
                        "url": "https://www.abortionfinder.org/",
                        "accessed_date": datetime.now().strftime('%Y-%m-%d')
                    }
                ]
                return {
                    "text": "To provide accurate abortion policy information, I need to know which U.S. state you're asking about. Could you please provide the state name or a ZIP code?",
                    "aspect_type": "policy",
                    "primary_content": "To provide accurate abortion policy information, I need to know which U.S. state you're asking about. Could you please provide the state name or a ZIP code?",
                    "question_answered": False,
                    "needs_state_info": True,
                    "citations": [c['source'] for c in default_citations],
                    "citation_objects": default_citations,
                    "processing_time": time.time() - start_time
                }

        except Exception as e:
            logger.error(f"Error processing policy query: {str(e)}", exc_info=True)
            # Use only AbortionFinder citation for error fallback
            fallback_citations = [
                {
                    "id": "abortionfinder", 
                    "source": "AbortionFinder",
                    "title": "Find Verified Abortion Care", 
                    "url": "https://www.abortionfinder.org/",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ]
            return {
                "text": f"I'm sorry, I'm having trouble processing your policy query. Please try again later or check the Guttmacher Institute website for accurate information.",
                "aspect_type": "policy",
                "primary_content": "",
                "question_answered": False,
                "needs_state_info": True, # Assume state info might be needed if error occurred
                "citations": [c['source'] for c in fallback_citations],
                "citation_objects": fallback_citations,
                "processing_time": time.time() - start_time
            }

    def _extract_structured_details(self, policy_data: Dict[str, Any], state_code: str) -> Dict[str, Any]:
        """Extracts structured policy details for the response composer."""
        details = {}
        
        # Check if we're dealing with error/fallback data
        has_error = policy_data.get("error", False)
        
        # If this is fallback or error data, provide minimal standardized details
        if has_error:
            logger.debug(f"Working with error/fallback data for {state_code}")
            # Set default legality from the abortion_legal field or fallback
            details['is_banned'] = not policy_data.get("abortion_legal", False)
            
            # Provide standardized fallback values for all expected detail sections
            details['gestational_limits'] = {
                'summary': 'Specific limit information unavailable',
                'banned': details['is_banned']
            }
            
            details['waiting_periods'] = {
                'summary': 'Waiting period information unavailable'
            }
            
            details['insurance_coverage'] = {
                'summary': 'Insurance coverage information unavailable'
            }
            
            details['minors'] = {
                'summary': 'Minors access information unavailable'
            }
            
            logger.debug(f"Created fallback structured details for {state_code}: {json.dumps(details)}")
            return details
        
        # Continue with normal processing for valid data
        endpoints = policy_data.get("endpoints", {})        
        logger.debug(f"Extracting structured details for {state_code} from endpoints: {', '.join(endpoints.keys())}")

        # Helper to safely get nested state data
        def get_state_data(endpoint_key):
            endpoint_data = endpoints.get(endpoint_key, {})
            logger.debug(f"Raw endpoint data for {endpoint_key}: {json.dumps(endpoint_data)[:200]}")
            
            # API structure varies: sometimes keyed by state, sometimes direct dict for the state
            if state_code in endpoint_data and isinstance(endpoint_data[state_code], dict):
                logger.debug(f"Found {endpoint_key} data directly keyed by state code")
                return endpoint_data[state_code]
            # Support for data keyed by state name (e.g., "New Mexico" instead of "NM")
            elif self.STATE_NAMES.get(state_code) in endpoint_data and isinstance(endpoint_data[self.STATE_NAMES.get(state_code)], dict):
                logger.debug(f"Found {endpoint_key} data keyed by state name")
                return endpoint_data[self.STATE_NAMES.get(state_code)]
            elif isinstance(endpoint_data, dict) and not any(k in self.STATE_NAMES for k in endpoint_data.keys()):
                 # If not keyed by state, assume data is directly for the requested state
                 # Check if 'state_code' field exists in the direct data and matches
                 if endpoint_data.get('state_code') == state_code or not endpoint_data.get('state_code'):
                     logger.debug(f"Using {endpoint_key} data as direct dict for state")
                     return endpoint_data
            return None # Return None if data structure is unexpected or doesn't match

        # Gestational Limits
        gestational = get_state_data("gestational_limits")
        logger.debug(f"Gestational data for {state_code}: {json.dumps(gestational) if gestational else None}")
        
        if gestational and isinstance(gestational, dict):
            details['gestational_limits'] = {}
            
            # Handle case where gestational data might be nested under state name
            if self.STATE_NAMES.get(state_code) in gestational:
                gestational = gestational[self.STATE_NAMES.get(state_code)]
                logger.debug(f"Extracted gestational data from state name key: {json.dumps(gestational)}")
                
            is_banned_flag = gestational.get("banned", False)
            
            # Handle explicit "no_restrictions" flag from API
            if gestational.get("no_restrictions", False):
                logger.debug(f"API indicates no restrictions for {state_code}")
                details['gestational_limits']['summary'] = "No specific gestational limit"
                details['gestational_limits']['banned'] = False
                details['is_banned'] = False
                
            # Check other standard fields
            elif is_banned_flag:
                logger.debug(f"API indicates abortion is banned in {state_code}")
                details['gestational_limits']['summary'] = "Banned with limited exceptions"
                details['gestational_limits']['banned'] = True
                details['is_banned'] = True
                
            else:
                weeks = gestational.get("banned_after_weeks_since_LMP")
                if weeks == 0 and weeks is not None:
                    details['gestational_limits']['summary'] = "Banned with limited exceptions"
                    details['gestational_limits']['banned'] = True
                    details['is_banned'] = True
                elif weeks == 99:
                    details['gestational_limits']['summary'] = "Legal until viability (typically around 24-26 weeks)"
                    details['gestational_limits']['banned'] = False
                    details['is_banned'] = False
                elif weeks is not None:
                    details['gestational_limits']['summary'] = f"Prohibited after {weeks} weeks LMP"
                    details['gestational_limits']['banned'] = False
                    details['is_banned'] = False
                elif gestational.get("no_specific_limit", False):
                     details['gestational_limits']['summary'] = "No specific gestational limit"
                     details['gestational_limits']['banned'] = False
                     details['is_banned'] = False
                else:
                     details['gestational_limits']['summary'] = "Specific limit information unavailable"
                     # Assume legal if not explicitly banned and no weeks given
                     details['gestational_limits']['banned'] = False
                     details['is_banned'] = False

        else:
             # Fallback legality check if gestational data is missing
             logger.debug(f"No gestational data found for {state_code}, using fallback legality check")
             details['is_banned'] = not self._is_abortion_legal(state_code) # Use internal check
             details['gestational_limits'] = {'summary': 'Specific limit information unavailable', 'banned': details['is_banned']}
             if details['is_banned']:
                  details['gestational_limits']['summary'] = "Banned with limited exceptions"

        # Waiting Periods
        waiting = get_state_data("waiting_periods")
        logger.debug(f"Waiting period data for {state_code}: {json.dumps(waiting) if waiting else None}")
        
        if waiting and isinstance(waiting, dict):
            # Handle case where waiting data might be nested under state name
            if self.STATE_NAMES.get(state_code) in waiting:
                waiting = waiting[self.STATE_NAMES.get(state_code)]
                logger.debug(f"Extracted waiting period data from state name key: {json.dumps(waiting)}")
            
            # If waiting_period_hours exists and is not None
            if waiting.get("waiting_period_hours") is not None:
                hours = waiting.get('waiting_period_hours', 0)
                if hours > 0:
                    details['waiting_periods'] = {
                        "summary": f"Required waiting period of {hours} hours",
                        "waiting_period_hours": hours
                    }
                else:
                     details['waiting_periods'] = {"summary": "No mandatory waiting period"}
            # Empty object likely means no waiting period
            elif waiting == {} or not waiting:
                details['waiting_periods'] = {"summary": "No mandatory waiting period"}
            else:
                details['waiting_periods'] = {"summary": "Waiting period information unavailable"}
        else:
             details['waiting_periods'] = {"summary": "Waiting period information unavailable"}


        # Insurance Coverage
        insurance = get_state_data("insurance_coverage")
        logger.debug(f"Insurance data for {state_code}: {json.dumps(insurance) if insurance else None}")
        
        if insurance and isinstance(insurance, dict):
            # Handle case where insurance data might be nested under state name
            if self.STATE_NAMES.get(state_code) in insurance:
                insurance = insurance[self.STATE_NAMES.get(state_code)]
                logger.debug(f"Extracted insurance data from state name key: {json.dumps(insurance)}")
                
            summary_parts = []
            if insurance.get("private_coverage_prohibited", False):
                 summary_parts.append("private insurance coverage is prohibited")
            if insurance.get("exchange_coverage_prohibited", False):
                 summary_parts.append("exchange plan coverage is prohibited")
            # Check for explicit "no restrictions" indicators  
            if insurance.get("private_coverage_no_restrictions", False) and insurance.get("exchange_coverage_no_restrictions", False):
                summary_parts = ["Coverage allowed with no restrictions"]
                
            medicaid_covered = insurance.get("medicaid_coverage_provides_for_abortion", False) or insurance.get("medicaid_coverage_provider", "") == "yes"
            if medicaid_covered:
                 summary_parts.append("Medicaid coverage is available in certain circumstances")

            if not summary_parts:
                details['insurance_coverage'] = {"summary": "Coverage allowed (restrictions may apply)"}
            else:
                details['insurance_coverage'] = {"summary": "; ".join(summary_parts).capitalize()}
        else:
             details['insurance_coverage'] = {"summary": "Insurance coverage information unavailable"}


        # Minors
        minors = get_state_data("minors")
        logger.debug(f"Minors data for {state_code}: {json.dumps(minors) if minors else None}")
        
        if minors and isinstance(minors, dict):
            # Handle case where minors data might be nested under state name
            if self.STATE_NAMES.get(state_code) in minors:
                minors = minors[self.STATE_NAMES.get(state_code)]
                logger.debug(f"Extracted minors data from state name key: {json.dumps(minors)}")
                
            summary_parts = []
            details['minors'] = {} # Ensure minors key exists
            
            # Check for explicit "allows_minor_to_consent_to_abortion" flag
            if minors.get("allows_minor_to_consent_to_abortion", False):
                summary_parts.append("No parental involvement required")
                details['minors']['parental_consent_required'] = False
                details['minors']['parental_notification_required'] = False
            elif minors.get("parental_consent_required", False):
                 summary_parts.append("Parental consent required")
                 details['minors']['parental_consent_required'] = True
            elif minors.get("parental_notification_required", False):
                 summary_parts.append("Parental notification required")
                 details['minors']['parental_notification_required'] = True
            else:
                 summary_parts.append("No parental involvement required")

            if minors.get('judicial_bypass_available', False):
                summary_parts.append("judicial bypass available")
                details['minors']['judicial_bypass_available'] = True

            details['minors']['summary'] = "; ".join(summary_parts)
        else:
            details['minors'] = {"summary": "Minors' access information unavailable"}

        # Log the final structured details
        logger.debug(f"Final structured details for {state_code}: {json.dumps(details)}")
        return details


    def _get_default_citation(self) -> Dict[str, Any]:
        """Returns a default citation object."""
        return {
            "id": "guttmacher-default",
            "source": "Guttmacher Institute",
            "title": "State Abortion Policy Landscape",
            "url": "https://www.guttmacher.org/state-policy",
            "accessed_date": datetime.now().strftime('%Y-%m-%d')
        }

    def _get_all_state_mentions(self, query: str, full_message: str = None) -> List[str]:
        """
        Get all state mentions in a query (for handling multi-state queries)

        Args:
            query (str): The query text
            full_message (str): The full message text for context

        Returns:
            List[str]: List of state codes found
        """
        message_text = full_message.lower() if full_message else query.lower()
        state_mentions = []

        # Check for greeting patterns that might trigger false "HI" detection
        has_greeting = False
        common_greetings = ["hi ", "hi,", "hi!", "hi abby", "hi there", "hello", "hey"]
        for greeting in common_greetings:
            if message_text.startswith(greeting):
                has_greeting = True
                logger.info(f"Detected greeting pattern '{greeting}' - will avoid false HI detection")
                break

        # --- IMPROVED: Prioritize full state names, especially multi-word ones ---
        # First look for full state names, matching longer ones first
        sorted_state_names = sorted(self.STATE_NAMES.items(), key=lambda item: len(item[1]), reverse=True)
        temp_message_text = message_text # Work on a temp copy for removals
        for state_code, state_name in sorted_state_names:
            # Ensure state_name is not empty or None before using regex
            if not state_name: continue
            state_pattern = r'\b' + re.escape(state_name.lower()) + r'\b'
            match_found = re.search(state_pattern, temp_message_text)
            if match_found:
                if state_code not in state_mentions: # Avoid duplicates from this step
                    state_mentions.append(state_code)
                    logger.info(f"Found state name mention: {state_name} ({state_code})")
                # Remove the matched state name to prevent partial matches later
                # Use count=1 to only remove the first instance found in this pass
                temp_message_text = re.sub(state_pattern, '', temp_message_text, count=1)


        # --- IMPROVED: Enhanced ambiguous abbreviation handling ---
        # Define context patterns for each ambiguous state code
        ambiguous_details = {
            'IN': {
                'keywords': ['indiana'], 
                'context': [r'state\s+of', r'laws\s+(in|for)', r'policy\s+(in|for)', r'abortion\s+(in|for)', 
                           r'live\s+in', r'from', r'in\s+(?=\d{5})'], 
                'prepositions': ['in'],
                'strong_indicators': [r'\bstate\s+of\s+in\b', r'\bindiana\b']
            },
            'ME': {
                'keywords': ['maine'], 
                'context': [r'state', r'laws', r'policy', r'abortion'], 
                'pronouns': ['me', 'my', 'i'],
                'strong_indicators': [r'\bmaine\b']
            },
            'OR': {
                'keywords': ['oregon'], 
                'context': [r'state', r'laws', r'policy', r'abortion'], 
                'conjunctions': [r'\w+\s+or\s+\w+', r'either\s+or', r'and/or'],
                'strong_indicators': [r'\boregon\b', r'\bstate\s+of\s+or\b']
            },
            'HI': {
                'keywords': ['hawaii'], 
                'context': [r'state', r'laws', r'policy', r'abortion'], 
                'greetings': common_greetings,
                'strong_indicators': [r'\bhawaii\b', r'\bstate\s+of\s+hi\b']
            },
            'PA': {
                'keywords': ['pennsylvania'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\bpennsylvania\b', r'\bstate\s+of\s+pa\b']
            },
            'LA': {
                'keywords': ['louisiana'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\blouisiana\b', r'\bstate\s+of\s+la\b']
            },
            'OH': {
                'keywords': ['ohio'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\bohio\b', r'\bstate\s+of\s+oh\b']
            },
            'OK': {
                'keywords': ['oklahoma'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\boklahoma\b', r'\bstate\s+of\s+ok\b']
            },
            'WA': {
                'keywords': ['washington'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\bwashington\b', r'\bstate\s+of\s+wa\b']
            },
            'DE': {
                'keywords': ['delaware'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\bdelaware\b', r'\bstate\s+of\s+de\b']
            },
            'CO': {
                'keywords': ['colorado'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\bcolorado\b', r'\bstate\s+of\s+co\b']
            },
            'VA': {
                'keywords': ['virginia'], 
                'context': [r'state', r'laws', 'policy', 'abortion'],
                'strong_indicators': [r'\bvirginia\b', r'\bstate\s+of\s+va\b']
            }
        }

        # Build specific pattern sets for common ambiguous cases
        state_names_lower_list = [state_name.lower() for state_name in self.STATE_NAMES.values() if state_name] 
        state_names_pattern = '|'.join([re.escape(state_name) for state_name in state_names_lower_list])
        
        # "OR" as conjunction patterns
        state_comparison_pattern = rf'({state_names_pattern})\s+or\s+({state_names_pattern})'
        or_conjunction_patterns = [ 
            r'\w+\s+or\s+\w+', 
            r'either\s+or',
            r'and/or',
            state_comparison_pattern 
        ]
        
        # "IN" as preposition patterns
        in_preposition_patterns = [ 
            rf'\bin\s+({state_names_pattern})', 
            r'\bin\s+a\s', 
            r'\bin\s+the\s', 
            r'\bin\s+my\s', 
            r'\bin\s+\d+', 
            r'live\s+in',
            r'abortion\s+in\s+'
        ]

        # Check for explicit uppercase state codes first (highest priority)
        pattern_upper = r'\b([A-Z]{2})\b'
        for match in re.finditer(pattern_upper, full_message): # Check original case message
            abbr = match.group(1)
            if abbr in self.STATE_NAMES and abbr not in state_mentions:
                state_mentions.append(abbr)
                logger.info(f"Found uppercase state code mention: {abbr}")

        # Check for lowercase state codes, being careful with ambiguous ones
        pattern_lower = r'\b([a-z]{2})\b'
        # Use the temp_message_text which has full names removed
        for match in re.finditer(pattern_lower, temp_message_text):
            abbr_lower = match.group(1)
            abbr_upper = abbr_lower.upper()

            if abbr_upper in self.STATE_NAMES and abbr_upper not in state_mentions:
                is_ambiguous = abbr_lower in AMBIGUOUS_ABBREVIATIONS
                skip = False

                if is_ambiguous:
                    details = ambiguous_details.get(abbr_upper, {})
                    # Get context window from the *original* lowered message text
                    context_window = message_text[max(0, match.start(1)-20):min(len(message_text), match.end(1)+20)]

                    # Check for strong indicators first (explicit state mentions)
                    has_strong_indicator = False
                    if 'strong_indicators' in details:
                        for indicator in details['strong_indicators']:
                            if re.search(indicator, message_text, re.IGNORECASE):
                                has_strong_indicator = True
                                logger.info(f"Found strong indicator for state {abbr_upper}: {indicator}")
                                break
                    
                    if has_strong_indicator:
                        # Strong indicators override ambiguity
                        skip = False
                    else:
                        # Rule out based on context (prepositions, conjunctions, greetings, pronouns)
                        if abbr_upper == 'HI' and has_greeting: 
                            skip = True
                            logger.info(f"Skipping 'hi' due to greeting pattern")
                        elif abbr_upper == 'OR' and any(re.search(p, context_window, re.IGNORECASE) for p in or_conjunction_patterns): 
                            skip = True
                            logger.info(f"Skipping 'or' as it appears to be used as a conjunction")
                        elif abbr_upper == 'IN' and any(re.search(p, context_window, re.IGNORECASE) for p in in_preposition_patterns): 
                            skip = True
                            logger.info(f"Skipping 'in' as it appears to be used as a preposition")
                        elif details and 'pronouns' in details and any(re.search(r'\b' + p + r'\b', context_window) for p in details['pronouns']): 
                            skip = True
                            logger.info(f"Skipping '{abbr_lower}' due to pronoun conflict")

                        # If not ruled out yet, check for positive context
                        if not skip and details and 'context' in details:
                             # Positive context requires specific keywords nearby
                             context_patterns = details["context"]
                             positive_context_pattern = rf'\b(?:{"|".join(context_patterns)})\s+{abbr_lower}\b|\b{abbr_lower}\s+(?:{"|".join(context_patterns)})\b'
                             has_positive_context = re.search(positive_context_pattern, context_window, re.IGNORECASE)
                             
                             # Special case for "state of XX"
                             state_of_pattern = rf'\bstate\s+of\s+{abbr_lower}\b'
                             has_state_of = re.search(state_of_pattern, context_window, re.IGNORECASE)
                             
                             if not (has_positive_context or has_state_of):
                                 skip = True
                                 logger.info(f"Skipping ambiguous code '{abbr_lower}' due to lack of positive context")

                if not skip:
                    state_mentions.append(abbr_upper)
                    context_desc = "" if not is_ambiguous else " (with sufficient context)"
                    logger.info(f"Found {'ambiguous ' if is_ambiguous else ''}state code mention: {abbr_upper}{context_desc}")
                else:
                     logger.info(f"Skipping ambiguous code '{abbr_lower}' due to context or lack thereof")


        # Look for state variants in separate method to keep code organized
        variant_states = self._check_state_variants(temp_message_text) # Use text with full names removed
        for state_code in variant_states:
            if state_code not in state_mentions:
                state_mentions.append(state_code)

        # Look for ZIP codes
        zip_matches = re.findall(r'\b(\d{5})\b', full_message) # Check original message for ZIP
        for zip_code in zip_matches:
            state_from_zip = self._get_state_from_zip(zip_code) # Use direct zip code here
            if state_from_zip and state_from_zip not in state_mentions:
                state_mentions.append(state_from_zip)
                logger.info(f"Found state from ZIP code {zip_code}: {state_from_zip}")

        # --- IMPROVED: Check for non-US countries AFTER US states/ZIPs ---
        non_us_countries = ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany',
                            'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']
        for country in non_us_countries:
            # Ensure it's a whole word match
             country_pattern = r'\b' + re.escape(country) + r'\b'
             # Avoid matching "mexico" within "new mexico"
             if country == 'mexico' and 'new mexico' in message_text:
                  continue
             if re.search(country_pattern, message_text):
                # Only log if NO US states were mentioned, otherwise US context takes precedence
                if not state_mentions:
                    logger.info(f"Non-US country detected and no US states found: {country}")
                    # If a non-US country is detected and no US states, clear mentions to avoid policy lookup
                    state_mentions = []
                    break # Stop checking countries if one is found without US states

        # Remove duplicates while preserving order
        unique_states = []
        for state in state_mentions:
            if state not in unique_states:
                unique_states.append(state)

        if not unique_states:
            logger.debug(f"No states found in message: '{message_text[:100]}...'")
        else:
            logger.info(f"Found {len(unique_states)} states in message: {', '.join(unique_states)}")

        return unique_states

    def _check_state_variants(self, text: str) -> List[str]:
        """
        Check for state name variants in text

        Args:
            text (str): Text to check (assumed lowercase)

        Returns:
            List[str]: List of state codes found
        """
        # Handle special cases for state name variations
        # Prioritize longer variants first
        state_variants = {
            # Common abbreviated forms
            "cali": "CA",
            "calif": "CA",
            "fla": "FL",
            "mass": "MA",
            "penn": "PA",
            "indy": "IN",
            "wash": "WA", # Ambiguous with DC, full name check is better
            "tex": "TX", 
            # Common informal/alternate names
            "socal": "CA", # Southern California
            "norcal": "CA", # Northern California
            "bay area": "CA",
            "upstate ny": "NY",
            "silicon valley": "CA",
            "twin cities": "MN",
            "garden state": "NJ",
            "empire state": "NY",
            "lone star": "TX",
            "sunshine state": "FL",
            "golden state": "CA",
            # Standard abbreviations (already handled in main method but backup here)
            "ny": "NY",
            "nj": "NJ",
            "az": "AZ",
            "oh": "OH",
            "ok": "OK",
            "pa": "PA",
            "va": "VA",
            "ca": "CA",
            "fl": "FL",
            "tx": "TX",
            "il": "IL",
            "mi": "MI",
            # Add more variants as needed
        }

        found_states = []
        processed_text = text # Use a copy to track removals

        # Check multi-word variants first, then single word variants
        multi_word_variants = {k: v for k, v in state_variants.items() if ' ' in k}
        single_word_variants = {k: v for k, v in state_variants.items() if ' ' not in k}
        
        # First check multi-word variants
        for variant, code in multi_word_variants.items():
            pattern = r'\b' + re.escape(variant) + r'\b'
            match_found = re.search(pattern, processed_text)
            if match_found:
                if code not in found_states:
                    found_states.append(code)
                    logger.info(f"Found multi-word state variant '{variant}' in message, matching to {code}")
                    # Remove the matched variant
                    processed_text = re.sub(pattern, '', processed_text, count=1)
        
        # Then check single-word variants
        for variant, code in single_word_variants.items():
            pattern = r'\b' + re.escape(variant) + r'\b'
            match_found = re.search(pattern, processed_text)
            if match_found:
                # Skip ambiguous variants if they are common words, unless context is strong
                is_ambiguous_variant = variant in AMBIGUOUS_ABBREVIATIONS
                if is_ambiguous_variant:
                     # Re-check context specifically for this variant match
                     context_window = text[max(0, match_found.start()-20):min(len(text), match_found.end()+20)]
                     
                     # Look for state-related context
                     context_pattern = rf'(?:state|laws|policy|abortion|access|in|for|near)\s+{variant}\b|\b{variant}\s+(?:state|laws|policy|abortion|access)'
                     state_of_pattern = rf'state\s+of\s+{variant}\b'
                     
                     has_context = re.search(context_pattern, context_window, re.IGNORECASE)
                     has_state_of = re.search(state_of_pattern, context_window, re.IGNORECASE)
                     
                     if not (has_context or has_state_of):
                          logger.info(f"Skipping ambiguous variant '{variant}' due to lack of context")
                          continue # Skip if ambiguous variant lacks context

                if code not in found_states:
                    found_states.append(code)
                    logger.info(f"Found state variant '{variant}' in message, matching to {code}")
                    # Remove the matched variant
                    processed_text = re.sub(pattern, '', processed_text, count=1)

        return found_states

    async def _handle_state_comparison(self, query: str, state_codes: List[str], full_message: str = None) -> Dict[str, Any]:
        """
        Handle comparison between multiple states

        Args:
            query (str): The query text
            state_codes (List[str]): List of state codes to compare
            full_message (str): The full message text

        Returns:
            Dict[str, Any]: Response with policy comparison
        """
        start_time = time.time()
        logger.info(f"Handling comparison for states: {state_codes}")

        try:
            # Limit to max 3 states to avoid overload
            if len(state_codes) > 3:
                logger.info(f"Limiting comparison from {len(state_codes)} states to first 3")
                state_codes = state_codes[:3]

            # Fetch policy data for all states
            policy_data_dict = {}
            fetch_tasks = [self._fetch_policy_data(state_code) for state_code in state_codes]
            results = await asyncio.gather(*fetch_tasks)

            for state_code, policy_data in zip(state_codes, results):
                 if not policy_data:
                     # If fetch failed, use fallback data
                     policy_data = self._get_fallback_policy_data(state_code)
                     logger.info(f"Using fallback data for {state_code} in comparison (API fetch failed)")
                 elif policy_data.get("error"):
                     logger.info(f"Using error-flagged data for {state_code} in comparison")
                 else:
                     logger.info(f"Using successfully fetched data for {state_code} in comparison")
                 
                 policy_data_dict[state_code] = policy_data


            # Prepare state names for better readability
            state_names_list = [f"{self.STATE_NAMES.get(code, code)}" for code in state_codes]
            state_names_str = ", ".join(state_names_list[:-1]) + f" and {state_names_list[-1]}" if len(state_names_list) > 1 else state_names_list[0]

            # Generate comparison text
            if self.openai_client:
                comparison_prompt = f"""Compare abortion laws and policies between {state_names_str}. Focus on key differences in legality, gestational limits, waiting periods, and parental consent/notification requirements for minors.

Start with a brief overview paragraph comparing the general restrictiveness.

Then, structure the comparison using the following format EXACTLY:

###Gestational Limits###
{state_names_list[0]}: [Details for {state_names_list[0]}]
"""
                # Add other states for Gestational Limits section
                for state_name in state_names_list[1:]:
                    comparison_prompt += f"{state_name}: [Details for {state_name}]\n"
                
                comparison_prompt += """
###Waiting Period###
"""
                # Add all states for Waiting Period section
                for state_name in state_names_list:
                    comparison_prompt += f"{state_name}: [Details for {state_name}]\n"
                
                comparison_prompt += """
###Insurance Coverage###
"""
                # Add all states for Insurance Coverage section
                for state_name in state_names_list:
                    comparison_prompt += f"{state_name}: [Details for {state_name}]\n"
                
                comparison_prompt += """
###Minors Access###
"""
                # Add all states for Minors Access section
                for state_name in state_names_list:
                    comparison_prompt += f"{state_name}: [Details for {state_name}]\n"
                
                comparison_prompt += """\n
IMPORTANT FORMATTING INSTRUCTIONS:
1. Use EXACTLY the section headers with ### delimiters as shown above (###Gestational Limits###, ###Waiting Period###, etc.)
2. DO NOT use ALL CAPS for headings or **bold** formatting for section titles
3. If information for a state is missing or unavailable, explicitly state 'Information unavailable' for that state in each section.
4. Do not omit any state from any section.
5. Insert a blank line after each section.

--- Policy Information Used ---

"""

                # Add structured policy data for each state
                for state_code in state_codes:
                    policy_data = policy_data_dict[state_code]
                    state_name = self.STATE_NAMES.get(state_code, state_code)
                    comparison_prompt += f"### {state_name} ({state_code}) Policy Summary:\n"
                    
                    # Check if this is error/fallback data
                    has_error = policy_data.get("error", False)
                    if has_error:
                        comparison_prompt += "- NOTE: Detailed policy information is currently unavailable for this state.\n"
                    
                    details = self._extract_structured_details(policy_data, state_code)
                    if details.get('is_banned'):
                        comparison_prompt += "- Legality: Banned with limited exceptions.\n"
                    else:
                        comparison_prompt += "- Legality: Generally legal.\n"
                    
                    # Add available details, or indicate unavailability
                    if 'gestational_limits' in details and details['gestational_limits'].get('summary'):
                        comparison_prompt += f"- Gestational Limit: {details['gestational_limits']['summary']}\n"
                    else:
                        comparison_prompt += "- Gestational Limit: Information not available\n"
                        
                    if 'waiting_periods' in details and details['waiting_periods'].get('summary'):
                        comparison_prompt += f"- Waiting Period: {details['waiting_periods']['summary']}\n"
                    else:
                        comparison_prompt += "- Waiting Period: Information not available\n"
                        
                    if 'minors' in details and details['minors'].get('summary'):
                         comparison_prompt += f"- Minors: {details['minors']['summary']}\n"
                    else:
                        comparison_prompt += "- Minors: Information not available\n"
                        
                    if 'insurance_coverage' in details and details['insurance_coverage'].get('summary'):
                        comparison_prompt += f"- Insurance: {details['insurance_coverage']['summary']}\n"
                    else:
                        comparison_prompt += "- Insurance: Information not available\n"
                    
                    comparison_prompt += "\n" # Add separator

                # Generate the comparison using OpenAI
                messages = [
                    {"role": "system", "content": "You are a helpful assistant providing accurate, concise comparisons of U.S. state abortion policies. Format your response with section headers using the ###SECTION NAME### format exactly as specified in the prompt. Always include all states in each section, even if information is unavailable for some states. DO NOT use ALL CAPS headings or **bold** formatting for section titles. Use only the exact ###Heading### format shown in the prompt."},
                    {"role": "user", "content": comparison_prompt}
                ]

                response_openai = await self.openai_client.chat.completions.create(
                    model=self.openai_model, # Use the configured model
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )

                comparison_text = response_openai.choices[0].message.content.strip()
            else:
                # Fallback without OpenAI - basic structured comparison
                comparison_text = f"Comparison of abortion policies for {state_names_str}:\n\n"

                # Start with a legality overview
                overview_parts = []
                for state_code in state_codes:
                    state_name = self.STATE_NAMES.get(state_code, state_code)
                    details = self._extract_structured_details(policy_data_dict[state_code], state_code)
                    status = 'has banned abortion with limited exceptions' if details.get('is_banned') else 'allows abortion with some regulations'
                    overview_parts.append(f"{state_name} {status}")
                
                comparison_text += "In this comparison: " + ". ".join(overview_parts) + ".\n\n"

                # Define the key details to extract and compare
                key_details = {
                    'Gestational Limits': lambda d: d.get('gestational_limits', {}).get('summary', 'Information not available'),
                    'Waiting Period': lambda d: d.get('waiting_periods', {}).get('summary', 'Information not available'),
                    'Insurance Coverage': lambda d: d.get('insurance_coverage', {}).get('summary', 'Information not available'),
                    'Minors Access': lambda d: d.get('minors', {}).get('summary', 'Information not available')
                }

                # Generate a standardized section for each key detail
                for detail_name, extractor in key_details.items():
                    comparison_text += f"###{detail_name}###\n"
                    for state_code in state_codes:
                        state_name = self.STATE_NAMES.get(state_code, state_code)
                        details = self._extract_structured_details(policy_data_dict[state_code], state_code)
                        value = extractor(details)
                        comparison_text += f"{state_name}: {value}\n"
                    comparison_text += "\n" # Add blank line after each section


            # Combine citations from all states
            all_citations = self._get_comparison_citation_objects(state_codes)
            citation_sources = list(set(c['source'] for c in all_citations)) # Unique source names

            # Measure processing time
            processing_time = time.time() - start_time
            logger.info(f"Policy comparison processed in {processing_time:.2f}s")

            # Prepare response object
            response = {
                "text": comparison_text,
                "primary_content": comparison_text, # Ensure primary content is set
                "aspect_type": "policy",
                "state_codes": state_codes,
                "question_answered": True,
                "needs_state_info": False,
                "confidence": 0.9,
                "citations": citation_sources,
                "citation_objects": all_citations,
                "processing_time": processing_time
            }

            return response

        except Exception as e:
            logger.error(f"Error processing policy comparison: {str(e)}", exc_info=True)
            default_citation = self._get_default_citation()
            state_names_display = [self.STATE_NAMES.get(code, code) for code in state_codes]
            return {
                "text": f"I'm sorry, I'm having trouble comparing abortion policies between {', '.join(state_names_display)}. Abortion laws vary by state and may change. You might consider checking the Guttmacher Institute website for the most current details.",
                "aspect_type": "policy",
                "primary_content": "",
                "question_answered": False,
                "needs_state_info": True, # Assume state info might be needed if error occurred
                "confidence": 0.5,
                "citations": [default_citation['source']],
                "citation_objects": [default_citation],
                 "processing_time": time.time() - start_time
            }

    def _get_state_from_query(self, query: str) -> Optional[str]:
        """
        Extract state name or code from the query, prioritizing full names.

        Args:
            query (str): User query text

        Returns:
            Optional[str]: Two-letter state code or None if not found
        """
        if not query or not isinstance(query, str): 
            logger.debug("_get_state_from_query received None or non-string query")
            return None
            
        query_lower = query.lower().strip()
        if not query_lower: 
            logger.debug("_get_state_from_query received empty query after stripping")
            return None

        logger.debug(f"_get_state_from_query analyzing query: '{query_lower}'")
        
        # --- IMPROVED: Special check for Hawaii at the end with context ---
        # Look for patterns like "policy in hi" where Hawaii is at the end
        hi_with_context_pattern = r'(?:policy|laws?|abortion|tell\s+me\s+about)\s+(?:in|for|of)\s+hi\b'
        hi_match = re.search(hi_with_context_pattern, query_lower)
        if hi_match and hi_match.end() > len(query_lower) - 5:  # Ensure it's near the end
            logger.info(f"Found specific context for Hawaii (HI) at end of query")
            return "HI"

        # --- IMPROVED: Prioritize full state names ---
        # Sort by length descending to match multi-word states first
        sorted_state_names = sorted(self.STATE_NAMES.items(), key=lambda item: len(item[1]), reverse=True)
        for state_code, state_name in sorted_state_names:
             if not state_name: continue # Skip if state name is empty
             # Use word boundaries for exact match
             state_pattern = r'\b' + re.escape(state_name.lower()) + r'\b'
             if re.search(state_pattern, query_lower):
                 logger.info(f"Found state name '{state_name}' ({state_code}) in query")
                 return state_code

        # --- Next, check for common state variants (Check BEFORE abbreviations) ---
        variant_states = self._check_state_variants(query_lower)
        if variant_states:
             # Return the first variant found
             logger.info(f"Found state variant matching '{variant_states[0]}' in query")
             return variant_states[0]

        # --- Last, check for state abbreviations with careful context analysis ---
        # First check for uppercase abbreviations (most reliable)
        for word in query.split(): # Use original case query here
             word_cleaned = re.sub(r'[^\w]', '', word) # Clean punctuation
             if len(word_cleaned) == 2 and word_cleaned.isupper() and word_cleaned in self.STATE_NAMES:
                  logger.info(f"Found uppercase state code '{word_cleaned}' in query")
                  return word_cleaned
        
        # Then try lowercase abbreviations with context checks
        words = query_lower.split()
        for word in words:
             word_cleaned = re.sub(r'[^\w]', '', word) # Clean punctuation
             if len(word_cleaned) == 2 and word_cleaned.lower() in self.STATE_NAMES_LOWER:
                  state_code = self.STATE_NAMES_LOWER[word_cleaned.lower()]
                  
                  # Extra context check for ambiguous abbreviations
                  if word_cleaned.lower() in AMBIGUOUS_ABBREVIATIONS:
                       # Special case for Hawaii (HI)
                       if word_cleaned.lower() == "hi":
                            # Check if it's a greeting at the beginning
                            is_greeting = query_lower.startswith("hi ")
                            # Check if it's the last word or has policy context
                            is_last_word = word == words[-1]
                            has_policy_context = "policy" in query_lower or "laws" in query_lower or "abortion" in query_lower
                            
                            if is_greeting and not (is_last_word and has_policy_context):
                                logger.info(f"Detected 'hi' is likely a greeting at start, not state code")
                                continue
                            elif is_last_word and has_policy_context:
                                logger.info(f"Detected 'hi' at end with policy context, likely Hawaii")
                                return state_code
                       
                       # For "in" check if used as preposition
                       if word_cleaned.lower() == "in" and (
                          " in " in query_lower and 
                          not any(phrase in query_lower for phrase in ["state of in", "indiana", "policies in", "laws in"])):
                            logger.info(f"Detected 'in' is likely used as a preposition, not state code")
                            continue
                            
                       # For "or" check if used as conjunction
                       if word_cleaned.lower() == "or" and (
                          " or " in query_lower and 
                          not any(phrase in query_lower for phrase in ["state of or", "oregon", "policies or", "laws or"])):
                            logger.info(f"Detected 'or' is likely used as a conjunction, not state code")
                            continue
                       
                       # Check if clear context supports this as a state code
                       context_phrases = ["state of", "laws in", "policy in", "abortion in"]
                       has_context = False
                       for phrase in context_phrases:
                            if f"{phrase} {word_cleaned.lower()}" in query_lower:
                                has_context = True
                                break
                                
                       if not has_context:
                            context_window = query_lower
                            ambiguous_pattern = r'\b(state|laws|policy|abortion)\s+(in|of|for)\s+' + re.escape(word_cleaned.lower()) + r'\b'
                            if not re.search(ambiguous_pattern, context_window):
                                 logger.info(f"Detected ambiguous '{word_cleaned}' lacks sufficient context as state code")
                                 continue
                                 
                  logger.info(f"Found state code '{state_code}' from '{word_cleaned}' in query")
                  return state_code
        
        # --- Last resort check: look for ZIP code and map to state ---
        zip_match = re.search(r'\b(\d{5})\b', query)
        if zip_match:
            zip_code = zip_match.group(1)
            state_from_zip = self._get_state_from_zip(zip_code)
            if state_from_zip:
                logger.info(f"Found state {state_from_zip} from ZIP code {zip_code} in query")
                return state_from_zip

        logger.debug(f"No state detected in query: '{query}'")
        return None


    def _get_state_from_history(self, user_location: Optional[Dict[str, str]],
                               conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        Extract state code from user location or conversation history

        Args:
            user_location (Optional[Dict[str, str]]): User's location data
            conversation_history (Optional[List[Dict[str, Any]]]): Conversation history

        Returns:
            Optional[str]: State code or None
        """
        # 1. Try user location first if provided
        if user_location:
            state_code = user_location.get('state_code', '').upper() # Prefer state_code
            if state_code and len(state_code) == 2 and state_code in self.STATE_NAMES:
                logger.info(f"Found state code in user location: {state_code}")
                return state_code

            state_name = user_location.get('state', '')
            if state_name:
                code_from_name = self._get_state_from_query(state_name) # Use query logic for name/variant matching
                if code_from_name:
                     logger.info(f"Found state code from state name in user location: {code_from_name}")
                     return code_from_name

            zip_code = user_location.get('zip', '') or user_location.get('postal_code', '')
            if zip_code:
                state_from_zip = self._get_state_from_zip(zip_code)
                if state_from_zip:
                    logger.info(f"Found state from ZIP code in user location: {state_from_zip}")
                    return state_from_zip

        # 2. Check conversation history
        if conversation_history:
            # Look at the last 5 messages (user and bot)
            for msg in reversed(conversation_history[-5:]):
                content = msg.get('content', msg.get('message', ''))
                if not content or not isinstance(content, str): continue

                # Try extracting state from message content
                state_found = self._get_state_from_query(query=content)
                if state_found:
                    # If found, double-check it's not ambiguous without context in *this* message
                    is_ambiguous = state_found.lower() in AMBIGUOUS_ABBREVIATIONS
                    if is_ambiguous:
                         context_pattern = rf'(?:state|laws|policy|abortion|access|in|for|near)\s+{state_found.lower()}\b|\b{state_found.lower()}\s+(?:state|laws|policy|abortion|access)'
                         if re.search(context_pattern, content.lower()):
                              logger.info(f"Found state {state_found} in conversation history message with context.")
                              return state_found
                         else:
                              logger.debug(f"Found ambiguous state {state_found} in history message, but lacking context in that specific message. Continuing search.")
                              continue # Don't return ambiguous state from history without context in the message itself
                    else:
                         logger.info(f"Found state {state_found} in conversation history message.")
                         return state_found


                # Check for ZIP code in the message
                zip_code = self._get_zip_code_from_query(content)
                if zip_code:
                    state_from_zip = self._get_state_from_zip(zip_code)
                    if state_from_zip:
                         logger.info(f"Found state {state_from_zip} from ZIP in conversation history")
                         return state_from_zip

        # No state found in history or user location
        logger.debug("No definitive state context found in history or user location.")
        return None

    async def _fetch_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Fetch policy data from the Abortion Policy API

        Args:
            state_code (str): Two-letter state code

        Returns:
            Dict[str, Any]: Policy data or fallback data on error
        """
        try:
            # Use correct API key and headers format
            headers = {'token': self.abortion_policy_api_key}
            policy_info = {"endpoints": {}}
            api_base = self.api_base_url.rstrip('/')

            logger.debug(f"Fetching policy data for {state_code} from {api_base}")
            logger.debug(f"API Key present: {bool(self.abortion_policy_api_key)}")

            async with aiohttp.ClientSession() as session:
                tasks = []
                for key, endpoint in self.api_endpoints.items():
                    url = f"{api_base}/{endpoint}/states/{state_code}"
                    logger.debug(f"Adding task to fetch {url}")
                    tasks.append(self._fetch_endpoint(session, url, headers, key))

                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                logger.debug(f"Got {len(results)} results for API calls")
                all_successful = True
                for key, result in zip(self.api_endpoints.keys(), results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {key} for {state_code}: {str(result)}")
                        policy_info["endpoints"][key] = {} # Assign empty dict on error
                        all_successful = False
                    elif isinstance(result, dict): # Ensure result is a dictionary
                        logger.debug(f"API result for {key}: {json.dumps(result)[:200]}")
                        policy_info["endpoints"][key] = result
                    else:
                        logger.warning(f"Unexpected result type for {key} for {state_code}: {type(result)}")
                        policy_info["endpoints"][key] = {} # Assign empty dict for unexpected types
                        all_successful = False # Treat unexpected types as failure

                # If any endpoint failed, return fallback to avoid partial data issues
                if not all_successful:
                     logger.warning(f"Returning fallback data for {state_code} due to API fetch errors.")
                     return self._get_fallback_policy_data(state_code)


            if any(policy_info["endpoints"].values()): # Check if any endpoint returned data
                 # Add state information directly to the top level
                 policy_info["state_code"] = state_code
                 policy_info["state_name"] = self.STATE_NAMES.get(state_code, state_code)
                 logger.info(f"Successfully fetched policy data for {state_code}")
                 
                 # Nest the actual data under the state code within each endpoint for consistency
                 formatted_endpoints = {}
                 for key, data in policy_info["endpoints"].items():
                      logger.debug(f"Processing endpoint data for {key}: {json.dumps(data)[:200]}")
                      if data: # Only process if data is not empty
                           # API structure varies, sometimes data is directly keyed by state, sometimes not
                           # Standardize to always have state code as key if data exists
                           # Check if data is already keyed by state_code OR if it's a direct dict for the state
                           if state_code in data and isinstance(data[state_code], dict):
                                logger.debug(f"Found {key} data keyed by state code")
                                formatted_endpoints[key] = {state_code: data[state_code]}
                           elif isinstance(data, dict) and not any(k in self.STATE_NAMES for k in data.keys()):
                                # Assumes data is directly for the state if not keyed by another state code
                                logger.debug(f"Found {key} data not keyed by state code, assuming direct dict")
                                formatted_endpoints[key] = {state_code: data}
                           else:
                                logger.debug(f"Data for {key} didn't match expected structure")
                                formatted_endpoints[key] = {} # Keep empty if data format is unexpected
                      else:
                           formatted_endpoints[key] = {}
                 
                 policy_info["endpoints"] = formatted_endpoints
                 policy_info["error"] = False # Explicitly mark as not an error
                 
                 # Set abortion_legal status directly in the policy_data for consistent access
                 # This ensures all code that uses policy_data has access to this field
                 structured_details = self._extract_structured_details(policy_info, state_code)
                 if 'is_banned' in structured_details:
                     policy_info['abortion_legal'] = not structured_details['is_banned']
                     logger.info(f"Setting abortion_legal={policy_info['abortion_legal']} based on structured details for {state_code}")
                 
                 return policy_info
            else:
                logger.error(f"No policy data found for {state_code} despite successful API calls.")
                return self._get_fallback_policy_data(state_code)

        except Exception as e:
            logger.error(f"Error fetching policy data for {state_code}: {str(e)}", exc_info=True)
            return self._get_fallback_policy_data(state_code)

    async def _fetch_endpoint(self, session, url, headers, key):
        """
        Fetch data from a specific endpoint

        Args:
            session: aiohttp ClientSession
            url: API endpoint URL
            headers: Request headers
            key: Endpoint key

        Returns:
            dict: API response data or empty dict on error
        """
        try:
            # Increased timeout slightly
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    try:
                        # Check content type before parsing JSON
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/json' in content_type:
                             json_data = await response.json()
                             # Handle potential empty responses from API which are valid JSON ("{}")
                             # Also handle cases where API returns a list for a state, take the first element
                             if isinstance(json_data, list) and len(json_data) > 0:
                                  return json_data[0] if isinstance(json_data[0], dict) else {}
                             elif isinstance(json_data, dict):
                                  return json_data if json_data else {}
                             else:
                                 logger.warning(f"API returned unexpected JSON structure for {url}: {type(json_data)}")
                                 return {}

                        else:
                             logger.warning(f"API returned non-JSON response for {url}. Status: {response.status}. Content-Type: {content_type}")
                             return {}
                    except json.JSONDecodeError as json_err:
                         logger.error(f"Failed to decode JSON from {url}: {json_err}")
                         return {}
                # Handle specific error codes if needed
                elif response.status == 401:
                     logger.error(f"API key unauthorized for {url}. Check ABORTION_POLICY_API_KEY.")
                     raise Exception("API Key Unauthorized") # Raise exception to signal critical failure
                elif response.status == 404:
                     logger.warning(f"API endpoint not found (404) for {url}. State or endpoint may be invalid.")
                     return {} # Treat as no data found
                else:
                    logger.warning(f"API returned status {response.status} for {url}")
                    # Log response body for debugging if possible
                    try:
                         body = await response.text()
                         logger.warning(f"Response body for {url} (status {response.status}): {body[:200]}...")
                    except Exception:
                         pass
                    return {}
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching endpoint {key} from {url}")
            raise # Reraise timeout to be caught by the main fetcher
        except aiohttp.ClientError as client_err:
             logger.error(f"Client error fetching endpoint {key} from {url}: {client_err}")
             raise # Reraise client error
        except Exception as e:
            logger.error(f"Generic error fetching endpoint {key}: {str(e)}")
            raise # Reraise other exceptions


    def _get_fallback_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Get fallback policy data when API fails

        Args:
            state_code (str): Two-letter state code

        Returns:
            Dict[str, Any]: Fallback policy data
        """
        state_name = self.STATE_NAMES.get(state_code, state_code)
        logger.warning(f"Returning fallback policy data for {state_name} ({state_code})")
        
        # Use hardcoded list to determine abortion_legal status for fallback
        BANNED_OR_HEAVILY_RESTRICTED = {
            "AL", "AR", "ID", "IN", "KY", "LA", "MS", "MO", "ND", "OK",
            "SD", "TN", "TX", "UT", "WV", "WI",
            "AZ", "FL", "GA", "IA", "NE", "NC", "SC", "WY"
        }
        is_legal = state_code not in BANNED_OR_HEAVILY_RESTRICTED
        logger.info(f"Setting fallback abortion_legal={is_legal} for {state_code} based on hardcoded list")
        
        fallback_citation = self._get_default_citation()
        return {
            "state_code": state_code,
            "state_name": state_name,
            "error": True,
            "error_message": "Could not retrieve live policy data.",
            "abortion_legal": is_legal,  # Add the abortion_legal field to fallback data
            "endpoints": {}, # Ensure endpoints key exists but is empty
            "resources": [ # Keep resources list
                "Planned Parenthood",
                "National Abortion Federation",
                "INeedAnA.com"
            ],
            "sources": [fallback_citation] # Use default citation object
        }

    async def _generate_with_openai(self, query: str, state_code: str, policy_data: Dict[str, Any], 
                              travel_state_code: str = None, travel_policy_data: Dict[str, Any] = None) -> str:
        """
        Generate a response using OpenAI model based on policy data

        Args:
            query (str): User query text
            state_code (str): Two-letter state code
            policy_data (Dict[str, Any]): Policy data for the requested state
            travel_state_code (str, optional): Two-letter state code for travel recommendation
            travel_policy_data (Dict[str, Any], optional): Policy data for the travel state

        Returns:
            str: Generated response
        """
        try:
            if not self.openai_client:
                logger.warning("OpenAI client not available for policy generation")
                return self._format_policy_data(state_code, policy_data, travel_state_code, travel_policy_data)

            start_time = time.time()

            # Get essential policy information in a structured format
            formatted_policy = self._extract_structured_details(policy_data, state_code)
            state_name = self.STATE_NAMES.get(state_code, state_code)
            
            # Prepare travel state information if available
            travel_info = ""
            if travel_state_code and travel_policy_data and travel_state_code != state_code:
                travel_state_name = self.STATE_NAMES.get(travel_state_code, travel_state_code)
                travel_formatted_policy = self._extract_structured_details(travel_policy_data, travel_state_code)
                
                travel_info = f"""
                Since abortion access is limited in {state_name}, many people travel to nearby states like {travel_state_name}.
                
                {travel_state_name} ({travel_state_code}) abortion policy information:
                - Legal Status: {travel_formatted_policy.get('legal_status', 'Information not available')}
                - Gestational Limits: {travel_formatted_policy.get('gestational_limits', {}).get('summary', 'Information not available')}
                - Waiting Period: {travel_formatted_policy.get('waiting_periods', {}).get('summary', 'None required')}
                - Insurance Coverage: {travel_formatted_policy.get('insurance_coverage', {}).get('summary', 'Information not available')}
                - Minor Requirements: {travel_formatted_policy.get('minors', {}).get('summary', 'Information not available')}
                """

            # Define the system message
            system_message = f"""
            You are Abby, a helpful and compassionate abortion information chatbot. 
            Your goal is to provide accurate, clear information about abortion policies in {state_name} ({state_code}) ONLY.
            
            Important formatting guidelines:
            1. Be factual, direct, and non-judgmental
            2. Start with a brief overview paragraph about abortion policy in {state_name}
            3. Structure the response with sections delimited EXACTLY like this:
               ###Gestational Limits###
               [Details for gestational limits]
               
               ###Waiting Period###
               [Details for waiting period]
               
               ###Insurance Coverage###
               [Details for insurance coverage]
               
               ###Minors Access###
               [Details for minors access]
               
            4. ONLY add a ###Travel Suggestion### section if travel information was provided
            5. DO NOT use ALL CAPS for headings or **bold** formatting for section titles
            6. Use EXACTLY the section names shown above with the ### delimiters
            7. Respond in a conversational but professional tone
            8. Always be supportive and compassionate
            9. Only mention {state_name} and {travel_state_name if travel_state_code else "NO OTHER STATE"}
            
            Here is the policy information to use in your response:
            
            {state_name} ({state_code}) abortion policy information:
            - Legal Status: {formatted_policy.get('legal_status', 'Information not available')}
            - Gestational Limits: {formatted_policy.get('gestational_limits', {}).get('summary', 'Information not available')}
            - Waiting Period: {formatted_policy.get('waiting_periods', {}).get('summary', 'None required')}
            - Insurance Coverage: {formatted_policy.get('insurance_coverage', {}).get('summary', 'Information not available')}
            - Minor Requirements: {formatted_policy.get('minors', {}).get('summary', 'Information not available')}
            
            {travel_info}
            
            Respond to the query with a brief overview paragraph followed by information in the required ###Section Name### format.
            """

            user_message = f"The user wants to know about abortion policies in {state_name}. Their query is: {query}"

            # Add debug logging to verify the prompt
            logger.debug(f"OpenAI Prompt for {state_code}:\nSYSTEM: {system_message}\nUSER: {user_message}")

            # Make API call to OpenAI
            completion = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=800
            )

            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Generated OpenAI response for {state_code} in {time.time() - start_time:.2f}s")
            return response_text

        except Exception as e:
            logger.error(f"OpenAI error for policy response: {str(e)}")
            # Fall back to template-based response
            return self._format_policy_data(state_code, policy_data, travel_state_code, travel_policy_data)

    def _format_policy_data(self, state_code: str, policy_data: Dict[str, Any],
                            travel_state_code: str = None, travel_policy_data: Dict[str, Any] = None) -> str:
        """
        Format policy data into a readable response

        Args:
            state_code (str): State code
            policy_data (Dict[str, Any]): Policy data dict
            travel_state_code (str, optional): Two-letter state code for travel recommendation
            travel_policy_data (Dict[str, Any], optional): Policy data for the travel state

        Returns:
            str: Formatted response
        """
        try:
            # Extract structured details for the main state
            details = self._extract_structured_details(policy_data, state_code)
            state_name = self.STATE_NAMES.get(state_code, state_code)
            
            # Start with main state information
            lines = []
            
            # Opening sentence
            if details.get('is_banned'):
                opener = f"In {state_name}, abortion is banned or highly restricted with limited exceptions."
            else:
                opener = f"In {state_name}, abortion is generally legal with some regulations."
                
            lines.append(opener)
            lines.append("")  # Empty line
            
            # Add details for main state with proper spacing
            if details.get('gestational_limits', {}).get('summary'):
                lines.append(f"###Gestational Limits###")
                lines.append(details['gestational_limits']['summary'])
                lines.append("")  # Empty line after content
                
            if details.get('waiting_periods', {}).get('summary'):
                lines.append(f"###Waiting Period###")
                lines.append(details['waiting_periods']['summary'])
                lines.append("")  # Empty line after content
                
            if details.get('insurance_coverage', {}).get('summary'):
                lines.append(f"###Insurance Coverage###")
                lines.append(details['insurance_coverage']['summary'])
                lines.append("")  # Empty line after content
                
            if details.get('minors', {}).get('summary'):
                lines.append(f"###Minors Access###")
                lines.append(details['minors']['summary'])
                lines.append("")  # Empty line after content
            
            # Add travel state information if provided
            if travel_state_code and travel_policy_data and travel_state_code != state_code:
                travel_details = self._extract_structured_details(travel_policy_data, travel_state_code)
                travel_state_name = self.STATE_NAMES.get(travel_state_code, travel_state_code)
                
                lines.append(f"###Travel Suggestion###")
                lines.append(f"Since abortion access is limited in {state_name}, many people travel to nearby states like {travel_state_name} for services.")
                
                if travel_details.get('gestational_limits', {}).get('summary'):
                    lines.append(f"In {travel_state_name}, abortion is available until {travel_details['gestational_limits']['summary'].lower()}")
                    
                if travel_details.get('waiting_periods', {}).get('summary'):
                    lines.append(f"Waiting period in {travel_state_name}: {travel_details['waiting_periods']['summary']}")
                    
                if travel_details.get('insurance_coverage', {}).get('summary'):
                    lines.append(f"Insurance in {travel_state_name}: {travel_details['insurance_coverage']['summary']}")
                
                lines.append("")  # Empty line after content
            
            # Closing line with a general note
            lines.append("Please note that abortion laws can change. For the most current information and personalized advice, consider consulting with a healthcare provider or contacting a reproductive health clinic.")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting policy data: {str(e)}")
            state_name = self.STATE_NAMES.get(state_code, state_code)
            return f"I have information about abortion policies in {state_name}, but I'm having trouble formatting it clearly. Please consider consulting AbortionFinder.org for up-to-date policy information."

    def _get_policy_citations(self, state_code: str) -> List[Dict[str, str]]:
        """
        Get citations for policy data

        Args:
            state_code (str): Two-letter state code

        Returns:
            List[Dict[str, str]]: List of citation objects
        """
        state_name = self.STATE_NAMES.get(state_code, state_code)
        # Use consistent base URL for Guttmacher state explorer
        guttmacher_url = "https://www.guttmacher.org/state-policy/explore/abortion-policy"
        pp_url = "https://www.plannedparenthood.org/learn/abortion/abortion-laws-by-state"
        accessed_date = datetime.now().strftime('%Y-%m-%d')

        citations = [
            {
                "id": f"guttmacher-{state_code.lower()}",
                "source": "Guttmacher Institute",
                "title": f"Abortion Policy in {state_name}",
                "url": guttmacher_url,
                "accessed_date": accessed_date
            },
             {
                "id": f"pp-{state_code.lower()}",
                "source": "Planned Parenthood",
                "title": f"Abortion Laws in {state_name}",
                "url": pp_url,
                "accessed_date": accessed_date
            }
        ]
        # Add AbortionFinder as a resource citation
        citations.append({
            "id": "abortionfinder",
            "source": "AbortionFinder",
            "title": "Find Verified Abortion Care",
            "url": "https://www.abortionfinder.org/",
            "accessed_date": accessed_date
        })


        return citations

    def _get_state_from_zip(self, zip_code: str) -> Optional[str]:
        """
        Extract state code from a ZIP code using available methods.

        Args:
            zip_code (str): The 5-digit ZIP code string.

        Returns:
            Optional[str]: State code or None.
        """
        if not zip_code or not re.fullmatch(r'\d{5}', zip_code):
            logger.warning(f"Invalid ZIP code format provided: {zip_code}")
            return None

        # 1. Try pyzipcode if available
        if ZipCodeDatabase:
            try:
                zcdb = ZipCodeDatabase()
                # pyzipcode might return multiple entries, get the first valid one
                zip_info_list = zcdb.find_zip(zip_code)
                if zip_info_list:
                    state = zip_info_list[0].state # Get state from the first ZipCode object
                    logger.info(f"Matched ZIP {zip_code} to state {state} using pyzipcode")
                    return state
            except Exception as e:
                logger.warning(f"pyzipcode lookup failed for {zip_code}: {e}")


        # 2. Fallback to internal range mapping
        logger.info(f"Using fallback ZIP range mapping for {zip_code}")
        return self._get_state_from_zip_fallback(zip_code)


    def _get_state_from_zip_fallback(self, zip_code: str) -> Optional[str]:
        """
        Fallback method using a simplified mapping of ZIP codes to states

        Args:
            zip_code (str): The 5-digit ZIP code string.

        Returns:
            Optional[str]: State code or None
        """
        if not zip_code or not re.fullmatch(r'\d{5}', zip_code):
            return None

        # Simple mapping of ZIP code ranges to states (ensure correct ranges)
        # Copied ranges from Preprocessor for consistency
        zip_ranges = {
            'AL': (35000, 36999), 'AK': (99500, 99999), 'AZ': (85000, 86999),
            'AR': (71600, 72999), 'CA': (90000, 96699), 'CO': (80000, 81999),
            'CT': (6000, 6999), 'DE': (19700, 19999), 'DC': (20000, 20599),
            'FL': (32000, 34999), 'GA': (30000, 31999), 'HI': (96700, 96899),
            'ID': (83200, 83999), 'IL': (60000, 62999), 'IN': (46000, 47999),
            'IA': (50000, 52999), 'KS': (66000, 67999), 'KY': (40000, 42799),
            'LA': (70000, 71599), 'ME': (3900, 4999), 'MD': (20600, 21999),
            'MA': (1000, 2799), 'MI': (48000, 49999), 'MN': (55000, 56999),
            'MS': (38600, 39999), 'MO': (63000, 65999), 'MT': (59000, 59999),
            'NE': (68000, 69999), 'NV': (89000, 89999), 'NH': (3000, 3899),
            'NJ': (7000, 8999), 'NM': (87000, 88499), 'NY': (10000, 14999), # Excludes 06390
            'NC': (27000, 28999), 'ND': (58000, 58999), 'OH': (43000, 45999),
            'OK': (73000, 74999), 'OR': (97000, 97999), 'PA': (15000, 19699),
            'RI': (2800, 2999), 'SC': (29000, 29999), 'SD': (57000, 57999),
            'TN': (37000, 38599), 'TX': (75000, 79999), # Excludes 73949, 885xx
            'UT': (84000, 84999),
            'VT': (5000, 5999), 'VA': (22000, 24699), # Excludes 201xx
            'WA': (98000, 99499),
            'WV': (24700, 26999), 'WI': (53000, 54999), 'WY': (82000, 83199)
        }
        # Handle specific exceptions before range checks
        if zip_code == '06390': return 'NY'
        if zip_code == '73949' or (zip_code.startswith('885')): return 'TX'
        if zip_code.startswith('201'): return 'VA'

        try:
            zip_int = int(zip_code)
            for state, (lower, upper) in zip_ranges.items():
                if lower <= zip_int <= upper:
                    # Refined checks for overlapping ranges if necessary (e.g., MA/NH/VT)
                    # if state == 'MA' and zip_int >= 3000: continue
                    # if state == 'NH' and zip_int >= 5000: continue
                    logger.info(f"Matched ZIP code {zip_code} to state {state} using fallback ranges")
                    return state
        except ValueError:
             logger.error(f"Could not convert zip code {zip_code} to integer for fallback check.")
             return None

        logger.warning(f"ZIP code {zip_code} did not match any fallback range.")
        return None

    def _get_supportive_resources_list(self, state_code: str) -> List[Dict[str, str]]:
        """
        Get a list of supportive resources for the specified state

        Args:
            state_code (str): Two-letter state code

        Returns:
            List[Dict[str, str]]: List of supportive resources
        """
        # This seems better placed in the ResponseComposer or a dedicated resource manager
        # Returning empty list here to avoid hardcoding in this handler.
        return []

    def _get_state_from_conversation(self, conversation_history: List[Dict]) -> Optional[str]:
        """
        Extract state information from conversation history (wrapper for _get_state_from_history).

        Args:
            conversation_history (List[Dict]): Previous conversation messages

        Returns:
            Optional[str]: State code if found, None otherwise
        """
        # This method essentially duplicates _get_state_from_history.
        # Recommend consolidating or calling the other method.
        return self._get_state_from_history(user_location=None, conversation_history=conversation_history)


    def _get_state_from_location(self, user_location: Dict[str, str]) -> Optional[str]:
        """
        Extract state code from user location data (wrapper for _get_state_from_history).

        Args:
            user_location (Dict[str, str]): User location data

        Returns:
            Optional[str]: State code if found, None otherwise
        """
        # This method essentially duplicates part of _get_state_from_history.
        # Recommend consolidating or calling the other method.
        return self._get_state_from_history(user_location=user_location, conversation_history=None)


    def _get_zip_code_from_query(self, query: str) -> Optional[str]:
        """
        Extract ZIP code from query text

        Args:
            query (str): Query text

        Returns:
            Optional[str]: 5-digit ZIP code if found, None otherwise
        """
        if not query or not isinstance(query, str):
            return None
        # Look for 5-digit ZIP code pattern using word boundaries
        match = re.search(r'\b(\d{5})\b', query)

        if match:
            zip_code = match.group(1)
            logger.debug(f"Extracted ZIP code {zip_code} from query.")
            return zip_code

        return None

    async def _find_nearby_clinics_async(self, zip_code: str, radius_miles: int = 50, 
                              state_context: str = None, is_travel_state: bool = False) -> Dict[str, Any]:
        """
        Async wrapper for finding abortion clinics near a given ZIP code
        
        Args:
            zip_code: ZIP code to search near
            radius_miles: Search radius in miles  
            state_context: Optional state code to focus search on
            is_travel_state: Whether this is a travel state different from the user's location
            
        Returns:
            Dictionary with clinic info and map center coordinates
        """
        try:
            # Handle travel state scenario - use a representative ZIP if needed
            if is_travel_state and state_context and not self._is_zip_in_state(zip_code, state_context):
                logger.info(f"ZIP {zip_code} is not in travel state {state_context}. Will use central location for state.")
                # For travel state searches where ZIP is in a different state, 
                # we should use a representative location in the travel state
                representative_zip = self._get_representative_zip_for_state(state_context)
                if representative_zip:
                    logger.info(f"Using representative ZIP {representative_zip} for travel state {state_context}")
                    zip_code = representative_zip
            
            # Use asyncio.to_thread to run the synchronous Google Maps API calls in a separate thread
            # This prevents blocking the event loop
            import asyncio
            result = await asyncio.to_thread(
                self._find_nearby_clinics, 
                zip_code=zip_code,
                radius_miles=radius_miles,
                state_context=state_context
            )
            return result
        except Exception as e:
            logger.error(f"Error in async clinic search for ZIP {zip_code}: {str(e)}")
            return {"clinics": [], "api_error": f"Async clinic search error: {str(e)}"}
    
    def _is_zip_in_state(self, zip_code: str, state_code: str) -> bool:
        """
        Check if a ZIP code is in the given state
        
        Args:
            zip_code: The ZIP code to check
            state_code: The state code to compare against
            
        Returns:
            bool: True if ZIP is in the state, False otherwise
        """
        if not zip_code or not re.fullmatch(r'\d{5}', zip_code):
            return False
        
        zip_state = self._get_state_from_zip(zip_code)
        return zip_state == state_code
    
    def _get_representative_zip_for_state(self, state_code: str) -> Optional[str]:
        """
        Get a representative ZIP code for a state, typically for its largest city or capital
        
        Args:
            state_code: The two-letter state code
            
        Returns:
            Optional[str]: A representative ZIP code for the state, or None if not found
        """
        # Map of representative ZIP codes for states (usually capital or largest city)
        representative_zips = {
            'AL': '36104', # Montgomery
            'AK': '99801', # Juneau
            'AZ': '85001', # Phoenix
            'AR': '72201', # Little Rock
            'CA': '94102', # San Francisco
            'CO': '80202', # Denver
            'CT': '06103', # Hartford
            'DE': '19901', # Dover
            'FL': '32801', # Orlando
            'GA': '30303', # Atlanta
            'HI': '96813', # Honolulu
            'ID': '83702', # Boise
            'IL': '60601', # Chicago
            'IN': '46204', # Indianapolis
            'IA': '50309', # Des Moines
            'KS': '66603', # Topeka
            'KY': '40202', # Louisville
            'LA': '70112', # New Orleans
            'ME': '04101', # Portland
            'MD': '21201', # Baltimore
            'MA': '02108', # Boston
            'MI': '48226', # Detroit
            'MN': '55401', # Minneapolis
            'MS': '39201', # Jackson
            'MO': '63101', # St. Louis
            'MT': '59601', # Helena
            'NE': '68508', # Lincoln
            'NV': '89101', # Las Vegas
            'NH': '03301', # Concord
            'NJ': '07102', # Newark
            'NM': '87501', # Santa Fe
            'NY': '10001', # New York
            'NC': '27601', # Raleigh
            'ND': '58501', # Bismarck
            'OH': '43215', # Columbus
            'OK': '73102', # Oklahoma City
            'OR': '97204', # Portland
            'PA': '19102', # Philadelphia
            'RI': '02903', # Providence
            'SC': '29201', # Columbia
            'SD': '57101', # Sioux Falls
            'TN': '37203', # Nashville
            'TX': '78701', # Austin
            'UT': '84111', # Salt Lake City
            'VT': '05401', # Burlington
            'VA': '23219', # Richmond
            'WA': '98101', # Seattle
            'WV': '25301', # Charleston
            'WI': '53202', # Milwaukee
            'WY': '82001', # Cheyenne
            'DC': '20001'  # Washington, D.C.
        }
        
        return representative_zips.get(state_code)

    def _find_nearby_clinics(self, zip_code: str, radius_miles: int = 50, state_context: str = None) -> Dict[str, Any]:
        """
        Find abortion clinics near a given ZIP code using Google Maps

        Args:
            zip_code: ZIP code to search near
            radius_miles: Search radius in miles
            state_context: Optional state code to focus search on (for travel state scenarios)

        Returns:
            Dictionary with "clinics" list and map center coordinates, or error information
        """
        # Ensure gmaps client is available
        if not self.gmaps:
            logger.warning("Google Maps client not initialized, cannot find clinics.")
            return {"clinics": [], "api_error": "Google Maps client not available"}

        # Validate ZIP code format
        if not zip_code or not re.fullmatch(r'\d{5}', zip_code):
            logger.warning(f"Invalid ZIP code format for clinic search: {zip_code}")
            return {"clinics": [], "api_error": "Invalid ZIP code format"}

        try:
            # Convert miles to meters for the API
            radius_meters = int(radius_miles * 1609.34)

            # 1. Geocode ZIP code
            try:
                geocode_result = self.gmaps.geocode(f"{zip_code}, USA")
                if not geocode_result or 'geometry' not in geocode_result[0] or 'location' not in geocode_result[0]['geometry']:
                    logger.warning(f"Could not geocode ZIP code {zip_code}")
                    return {"clinics": [], "api_error": "Could not geocode ZIP code"}
                location = geocode_result[0]['geometry']['location'] # {lat: float, lng: float}
                
                # Extract country and state information from geocode result
                zip_country = None
                zip_state = None
                for component in geocode_result[0].get('address_components', []):
                    if 'country' in component.get('types', []):
                        zip_country = component.get('short_name')
                    if 'administrative_area_level_1' in component.get('types', []):
                        zip_state = component.get('short_name')
                
                # Check if ZIP is in the US
                if zip_country and zip_country != 'US':
                    logger.warning(f"ZIP code {zip_code} appears to be outside the US: {zip_country}")
                    return {"clinics": [], "api_error": f"ZIP code appears to be in {zip_country}, not US"}
                
                # Check if ZIP state matches the state_context if provided
                if state_context and zip_state and zip_state != state_context:
                    logger.warning(f"ZIP code {zip_code} is in state {zip_state}, but search context is for {state_context}")
                    # Continue search but note the mismatch in log
                
                logger.info(f"Geocoded ZIP {zip_code} to: {location}, State: {zip_state}")
            except Exception as geo_e:
                error_msg = str(geo_e)
                logger.error(f"Error geocoding ZIP code {zip_code}: {error_msg}")
                if "REQUEST_DENIED" in error_msg or "API key" in error_msg or "not authorized" in error_msg:
                    return {"clinics": [], "api_error": f"API authorization error: {error_msg}"}
                return {"clinics": [], "api_error": f"Geocoding error: {error_msg}"}

            # Create result structure with center coordinates
            result = {
                "clinics": [],
                "center": {
                    "lat": location['lat'],
                    "lng": location['lng']
                },
                "origin_zip": zip_code,
                "origin_state": zip_state
            }
            
            # If we're checking a travel state but the ZIP is in a different state
            if state_context and zip_state and zip_state != state_context:
                # Add a flag in the result to indicate this mismatch
                result["zip_state_mismatch"] = True
                result["intended_state"] = state_context
                logger.warning(f"Continuing search with ZIP in different state (ZIP:{zip_state}, target:{state_context})")

            # 2. Search for relevant places
            # --- Modify search to incorporate state_context if provided ---
            search_terms = [
                "abortion clinic",
                "abortion services",
                "planned parenthood", # Often provides services
                "women's health clinic", # May provide services
                "family planning clinic" # May provide services
            ]
            
            # If state_context is provided, add it to some of the search terms
            if state_context:
                state_name = self.STATE_NAMES.get(state_context, state_context)
                # Add state-specific search terms - use state name for better results
                state_search_terms = [
                    f"abortion clinic in {state_name}",
                    f"planned parenthood {state_name}"
                ]
                # Insert at beginning to prioritize state-specific results
                search_terms = state_search_terms + search_terms
                logger.info(f"Added state-specific search terms for {state_context} ({state_name})")
            
            all_places = {} # Use dict with place_id as key for deduplication
            api_errors = []

            for term in search_terms:
                 try:
                     logger.info(f"Searching for '{term}' near {zip_code} ({location})")
                     # Use text_search for broader matching including name and type
                     results = self.gmaps.places_nearby(
                          location=(location['lat'], location['lng']),
                          radius=radius_meters,
                          keyword=term # Use keyword for relevance
                     )

                     if results.get('results'):
                          logger.info(f"Found {len(results['results'])} potential places for '{term}'")
                          for place in results['results']:
                               place_id = place.get('place_id')
                               if place_id and place_id not in all_places:
                                    all_places[place_id] = place
                     elif results.get('status') != 'OK' and results.get('status') != 'ZERO_RESULTS':
                          # Log API errors other than just no results
                          error_message = f"{results.get('status')} - {results.get('error_message', '')}"
                          logger.warning(f"Google Maps API status error for '{term}': {error_message}")
                          
                          # Check for specific API key errors
                          if results.get('status') == 'REQUEST_DENIED':
                               return {"clinics": [], "api_error": f"Google Maps API access denied: {results.get('error_message', 'API key may not be authorized for this service')}"}
                          
                          api_errors.append(error_message)

                 except Exception as search_e:
                     error_msg = str(search_e)
                     logger.error(f"Error during Google Maps places_nearby search for '{term}': {error_msg}")
                     if "REQUEST_DENIED" in error_msg or "API key" in error_msg or "not authorized" in error_msg:
                         return {"clinics": [], "api_error": f"API authorization error: {error_msg}"}
                     api_errors.append(error_msg)

            # Check for consistent API errors
            if api_errors and not all_places:
                # If we had errors for all search terms and found no places, report the error
                return {"clinics": [], "api_error": f"Google Maps API errors: {'; '.join(api_errors[:2])}"}

            if not all_places:
                logger.warning(f"No potential clinics found via nearby search for ZIP {zip_code}")
                return result

            # 3. Filter and enrich results
            clinics = []
            place_ids_to_fetch = list(all_places.keys())

            # Fetch details in batches if needed (though Places API doesn't batch easily)
            for place_id in place_ids_to_fetch:
                 original_place_data = all_places[place_id]
                 place_name = original_place_data.get('name', '')

                 # Filter out unlikely clinics based on name *before* fetching details
                 if not self._is_likely_clinic(place_name):
                      continue

                 # Fetch detailed info
                 try:
                      # IMPORTANT: Ensure 'types' is NOT in this list - it causes API errors
                      fields_to_request = ['name', 'formatted_address', 'formatted_phone_number', 
                                         'website', 'geometry/location', 'place_id', 'rating']
                      logger.debug(f"Requesting details for place '{place_name}' with fields: {fields_to_request}")
                      
                      details_response = self.gmaps.place(place_id, fields=fields_to_request)
                      details = details_response.get('result', {})

                      clinic_location = details.get('geometry', {}).get('location', {})
                      clinics.append({
                          "name": details.get('name', place_name),
                          "address": details.get('formatted_address', original_place_data.get('vicinity', '')),
                          "phone": details.get('formatted_phone_number', ''),
                          "website": details.get('website', ''),
                          "rating": details.get('rating', original_place_data.get('rating')),
                          "latitude": clinic_location.get('lat'),
                          "longitude": clinic_location.get('lng'),
                          "place_id": place_id
                      })
                 except Exception as detail_e:
                      logger.error(f"Error getting details for place {place_id} ({place_name}): {detail_e}")


            if not clinics:
                logger.warning(f"No likely clinics found after filtering for ZIP code {zip_code}")
                return result

            # 4. Sort results (e.g., by proximity or rating)
            # Handle None ratings during sort
            clinics.sort(key=lambda x: x.get('rating') if x.get('rating') is not None else -1.0, reverse=True)

            # Add top clinics to the result (maximum 5)
            result["clinics"] = clinics[:5]

            logger.info(f"Found {len(clinics)} likely clinics near {zip_code}.")
            return result

        except Exception as e:
            logger.error(f"General error finding clinics near {zip_code}: {str(e)}", exc_info=True)
            # Return default structure but include the error message
            return {"clinics": [], "api_error": f"Clinic search processing error: {str(e)}"}


    def _get_fallback_clinics_with_zip(self, zip_code: str) -> Dict[str, Any]:
        """
        Return fallback clinics for a zip code

        Args:
            zip_code (str): ZIP code

        Returns:
            Dictionary with empty clinics list
        """
        # Return an empty dictionary with clinics key to avoid showing hardcoded resources
        # Let the ResponseComposer handle fallback resources if needed
        logger.info(f"Using fallback (empty) for clinic search near {zip_code}")
        return {"clinics": []}

    def _is_likely_clinic(self, name: str) -> bool:
        """
        Check if a place name is likely to be an abortion provider.
        Improved to better filter Crisis Pregnancy Centers (CPCs).

        Args:
            name: Name of the place

        Returns:
            True if likely a clinic, False otherwise
        """
        if not name: return False
        name_lower = name.lower()

        # Keywords that indicate a place IS likely an abortion provider
        positive_indicators = [
            "planned parenthood",
            "family planning",
            "abortion services", # More specific
            "abortion care",   # More specific
            "reproductive health services", # Broader but relevant
            "women's health services", # Broader but relevant
            "women's center", # Often provide care
            "health center", # Can be ambiguous, lower priority
            "clinic", # Can be ambiguous, lower priority
            "surgical center", # Could be relevant if context fits
            "medical group" # Could be relevant if context fits
        ]

        # Keywords that indicate a place is NOT an abortion provider (Crisis Pregnancy Centers etc.)
        negative_indicators = [
            "crisis pregnancy", "cpc",
            "pregnancy resource", "prc"
            "pro life", # Split into two words
            "pro-life", # Keep hyphenated version
            "prolife", # Keep combined version
            "right to life",
            "pregnancy support",
            "pregnancy help",
            "adoption agency", "adoption services",
            "maternity home",
            "anti-abortion",
            "care net", # Known CPC network
            "heartbeat international", # Known CPC network
            "birthright", # Known CPC network
            "option line", # Known CPC network
            "real options", # Common CPC name pattern
            "alternatives", # Common CPC name pattern
            "life choices", # Common CPC name pattern
            "hope pregnancy", # Common CPC name pattern
            "first choice pregnancy", # Common CPC name pattern
            "vitae foundation",
            "focus on the family",
            "students for life",
            # Terms often associated with counseling *against* abortion
            "post-abortion counseling" # Can be offered by CPCs
            "options counseling" # Can be offered by CPCs, needs more context
        ]

        # Check for negative indicators first - if found, definitely exclude
        for indicator in negative_indicators:
             pattern = r'\b' + re.escape(indicator) + r'\b'
             if re.search(pattern, name_lower):
                 logger.debug(f"Excluding '{name}' due to negative indicator: '{indicator}'")
                 return False

        # Then check for positive indicators
        has_positive = False
        for indicator in positive_indicators:
             pattern = r'\b' + re.escape(indicator) + r'\b'
             if re.search(pattern, name_lower):
                  # Added check: Ensure it's not negated (e.g., "No abortion services")
                  negation_pattern = rf'\b(no|not|non|without|anti)\s+({re.escape(indicator)})\b'
                  if not re.search(negation_pattern, name_lower):
                       logger.debug(f"Including '{name}' due to positive indicator: '{indicator}'")
                       has_positive = True
                       break # Found a positive indicator

        if has_positive:
             # Optional: Add a final check for ambiguity, e.g., if "clinic" is present but also "resource center"
             # if "clinic" in name_lower and ("resource" in name_lower or "support" in name_lower):
             #      logger.debug(f"Ambiguous name '{name}' - positive and potential negative indicators. Excluding.")
             #      return False
             return True

        # If no strong indicators found, default to False (be conservative)
        logger.debug(f"Excluding '{name}' due to lack of clear positive indicators or presence of negative ones.")
        return False

    def _get_comparison_citation_objects(self, state_codes: List[str]) -> List[Dict[str, str]]:
        """
        Get citation objects for a comparison of multiple states

        Args:
            state_codes (List[str]): List of two-letter state codes

        Returns:
            List[Dict[str, str]]: List of citation objects
        """
        # Consistent URLs
        guttmacher_url = "https://www.guttmacher.org/state-policy/explore/abortion-policy"
        pp_url = "https://www.plannedparenthood.org/learn/abortion/abortion-laws-by-state"
        finder_url = "https://www.abortionfinder.org/"
        accessed_date = datetime.now().strftime('%Y-%m-%d')

        # Start with general comparison sources
        citations = [
            {
                "id": "guttmacher-comparison",
                "source": "Guttmacher Institute",
                "title": f"State Abortion Policy Comparison", # Generic title
                "url": guttmacher_url,
                "accessed_date": accessed_date
            },
            {
                "id": "pp-comparison",
                "source": "Planned Parenthood",
                "title": "Abortion Laws by State",
                "url": pp_url,
                "accessed_date": accessed_date
            },
             {
                "id": "abortionfinder-comparison",
                "source": "AbortionFinder",
                "title": "State Guides & Information",
                "url": finder_url, # Link to main site for comparison context
                "accessed_date": accessed_date
            }
        ]
        # Keep citations concise for comparison, avoid listing each state individually here

        return citations


    def _get_less_restrictive_states(self, state_code: str) -> List[str]:
        """
        Get nearby less restrictive states for travel recommendations.
        
        This method leverages the abortion_access utils if available, and falls back to 
        hardcoded data if necessary.

        Args:
            state_code (str): Two-letter state code for the restrictive state

        Returns:
            List[str]: List of nearby state codes that are less restrictive, sorted by proximity
        """
        logger.info(f"Finding nearby less restrictive states for {state_code}")
        
        # Try using the imported function first
        try:
            if get_less_restrictive_states and callable(get_less_restrictive_states):
                nearby_states = get_less_restrictive_states(state_code)
                if nearby_states:
                    logger.info(f"Found nearby less restrictive states using utility: {', '.join(nearby_states)}")
                    return nearby_states
        except Exception as e:
            logger.warning(f"Error using abortion_access.get_less_restrictive_states: {e}")
        
        # Fallback mapping of restrictive states to their closest less-restrictive neighbors
        # Ordered by proximity/travel convenience
        fallback_nearest_states = {
            # Banned/Highly Restricted States -> Nearby Legal States
            'AL': ['FL', 'NC', 'VA'],  # Alabama -> Florida, North Carolina, Virginia
            'AR': ['IL', 'KS', 'NM'],  # Arkansas -> Illinois, Kansas, New Mexico
            'AZ': ['CA', 'NV', 'NM'],  # Arizona -> California, Nevada, New Mexico
            'FL': ['NC', 'VA', 'DC'],  # Florida -> North Carolina, Virginia, DC
            'GA': ['NC', 'VA', 'DC'],  # Georgia -> North Carolina, Virginia, DC
            'IA': ['IL', 'MN', 'NE'],  # Iowa -> Illinois, Minnesota, Nebraska
            'ID': ['WA', 'OR', 'MT'],  # Idaho -> Washington, Oregon, Montana
            'IN': ['IL', 'MI', 'OH'],  # Indiana -> Illinois, Michigan, Ohio
            'KY': ['IL', 'VA', 'OH'],  # Kentucky -> Illinois, Virginia, Ohio
            'LA': ['IL', 'NM', 'FL'],  # Louisiana -> Illinois, New Mexico, Florida
            'MO': ['IL', 'KS', 'NE'],  # Missouri -> Illinois, Kansas, Nebraska
            'MS': ['IL', 'FL', 'NC'],  # Mississippi -> Illinois, Florida, North Carolina
            'MT': ['WA', 'OR', 'MN'],  # Montana -> Washington, Oregon, Minnesota
            'ND': ['MN', 'MT', 'SD'],  # North Dakota -> Minnesota, Montana, South Dakota
            'NE': ['IL', 'MN', 'CO'],  # Nebraska -> Illinois, Minnesota, Colorado
            'OH': ['MI', 'PA', 'VA'],  # Ohio -> Michigan, Pennsylvania, Virginia 
            'OK': ['KS', 'NM', 'CO'],  # Oklahoma -> Kansas, New Mexico, Colorado
            'SC': ['NC', 'VA', 'DC'],  # South Carolina -> North Carolina, Virginia, DC
            'SD': ['MN', 'MT', 'WY'],  # South Dakota -> Minnesota, Montana, Wyoming
            'TN': ['IL', 'VA', 'NC'],  # Tennessee -> Illinois, Virginia, North Carolina
            'TX': ['NM', 'KS', 'CO'],  # Texas -> New Mexico, Kansas, Colorado
            'UT': ['CO', 'NV', 'NM'],  # Utah -> Colorado, Nevada, New Mexico
            'WI': ['IL', 'MN', 'MI'],  # Wisconsin -> Illinois, Minnesota, Michigan
            'WV': ['PA', 'MD', 'VA'],  # West Virginia -> Pennsylvania, Maryland, Virginia
            'WY': ['CO', 'MT', 'NE'],  # Wyoming -> Colorado, Montana, Nebraska
        }
        
        # If state is in our fallback map, return its less restrictive neighbors
        if state_code in fallback_nearest_states:
            result = fallback_nearest_states[state_code]
            logger.info(f"Using fallback less restrictive states for {state_code}: {', '.join(result)}")
            return result
            
        # Last resort: return a default list of generally accessible states
        default_accessible = ['CA', 'NY', 'IL', 'MA', 'CO']
        logger.warning(f"No specific less restrictive states found for {state_code}, using defaults")
        return default_accessible

    def _is_abortion_legal(self, state_code: str, policy_data: Dict[str, Any] = None) -> bool:
        """
        Determine if abortion is generally legal in a state based on available information.
        
        This method uses a combination of policy API data and fallback lists to determine
        if abortion is generally legal in a given state.

        Args:
            state_code (str): Two-letter state code
            policy_data (Dict[str, Any], optional): Policy data if already fetched

        Returns:
            bool: True if abortion is generally legal, False if banned/heavily restricted
        """
        logger.info(f"Checking if abortion is legal in {state_code}")
        
        # Try using the imported function first
        try:
            if is_abortion_legal_in_state and callable(is_abortion_legal_in_state):
                is_legal = is_abortion_legal_in_state(state_code)
                logger.info(f"Abortion legality in {state_code} from utility: {is_legal}")
                return is_legal
        except Exception as e:
            logger.warning(f"Error using abortion_access.is_abortion_legal_in_state: {e}")
        
        # Try to determine from policy data if available
        if policy_data and not policy_data.get("error", True):
            # If we have actual policy data, try to determine from that
            gest_limits = policy_data.get("gestational_limits", {})
            if gest_limits:
                # Check if banned at all stages or extremely limited
                banned_indicators = ["banned", "prohibition", "not permitted", "illegal"]
                limit_text = str(gest_limits.get("summary", "")).lower()
                if any(indicator in limit_text for indicator in banned_indicators):
                    logger.info(f"Abortion appears restricted in {state_code} based on gestational limit text: '{limit_text}'")
                    return False
                
                # Check for reasonable gestational limits (generally > 6 weeks to be considered accessible)
                # Look for patterns like "X weeks" and extract the number
                import re
                week_match = re.search(r'(\d+)\s*weeks?', limit_text)
                if week_match:
                    try:
                        weeks = int(week_match.group(1))
                        if weeks <= 6:  # Very early limits are effectively bans
                            logger.info(f"Abortion in {state_code} has very restrictive limit of {weeks} weeks")
                            return False
                        else:
                            logger.info(f"Abortion in {state_code} appears legal with limit of {weeks} weeks")
                            return True
                    except (ValueError, IndexError):
                        pass  # Continue to fallback if we can't parse the weeks
        
        # Use the fallback of known banned/restricted states if we couldn't determine from policy data
        # This list needs to be updated periodically as laws change
        BANNED_OR_HEAVILY_RESTRICTED = [
            'AL', 'AR', 'ID', 'IN', 'KY', 'LA', 'MS', 'MO', 'ND', 'OK', 
            'SD', 'TN', 'TX', 'UT', 'WV', 'WI', 'WY', 'GA', 'IA', 'SC', 'NC', 'OH', 'MT'
        ]
        # Updated: 2025-04-13 - This list should be reviewed regularly

        result = state_code not in BANNED_OR_HEAVILY_RESTRICTED
        logger.info(f"Using fallback list to determine abortion legality in {state_code}: {result}")
        return result

# --- END OF FILE policy_handler.py ---