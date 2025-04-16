# --- START OF FILE conversation_manager.py ---

import logging
import time
import random
import uuid
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Define mock implementations for modules that don't exist in the codebase
# BaselineModel mock implementation
class BaselineModel:
    def __init__(self, evaluation_model="local"):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using BaselineModel implementation")
    
    def process_message(self, message, conversation_id=None, persona="baseline", history=None):
        return {
            "text": "I'm sorry, the baseline model is not available. Please try again later.",
            "citations": [],
            "citation_objects": []
        }

# FriendlyBot mock implementation
class FriendlyBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using FriendlyBot implementation")
        
    def process_message(self, message, context=None):
        return "I'm sorry, the friendly bot is not available. Please try again later."

# Import existing modules
try:
    from .citation_manager import CitationManager
except ImportError:
    # Create a minimal mock CitationManager if the real one is missing
    class CitationManager:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Using mock CitationManager implementation")
        
        def format_citations(self, citations):
            return citations

try:
    from .policy_api import PolicyAPI
except ImportError:
    # Create a minimal mock PolicyAPI if the real one is missing
    class PolicyAPI:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Using mock PolicyAPI implementation")
            self.STATE_NAMES = {}
        
        def get_policy_data(self, state_code):
            return {"error": True, "message": "Policy API is not available"}
            
        def _extract_state_from_question(self, question):
            return None

try:
    from .question_classifier import QuestionClassifier
except ImportError:
    # Create a minimal mock QuestionClassifier if the real one is missing
    class QuestionClassifier:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Using mock QuestionClassifier implementation")
        
        def classify(self, question):
            return {"category": "unknown", "confidence": 0.0}

# ContextManager mock implementation
class ContextManager:
    def __init__(self, max_context_length=5):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using ContextManager implementation")
        self.context = []
        self.max_context_length = max_context_length
    
    def add_message(self, message):
        self.context.append(message)
        if len(self.context) > self.max_context_length:
            self.context.pop(0)
    
    def get_context(self):
        return self.context.copy()
    
    def clear_context(self):
        self.context = []

# ResponseEvaluator mock implementation
class ResponseEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using ResponseEvaluator implementation")
    
    def evaluate(self, response, context):
        return {"score": 0.5, "feedback": "No evaluation available"}
    
    def get_metrics(self):
        return {"accuracy": 0.0, "empathy": 0.0, "helpfulness": 0.0}

import requests # Keep requests for potential external calls (like ZIP lookup fallback)
import re
try:
    import us # Keep for state name lookup if needed by _format_abortion_policy_response
except ImportError:
    us = None # Define us as None if import fails
import openai # Keep if BaselineModel uses it
try:
    from flask import session # Keep if session management is done here (though MemoryManager might handle it)
except ImportError:
    session = None # Define session as None if import fails
import nltk # Keep if BaselineModel or other components rely on it

# Import PII detector and language detection from utils
try:
    from utils.text_processing import PIIDetector, detect_language
    from utils.data_loader import load_reproductive_health_data
except ImportError:
    # Create fallback implementations
    logger = logging.getLogger(__name__)
    logger.warning("Failed to import utils modules. Using fallback implementations.")
    
    class PIIDetector:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            
        def redact_pii(self, text):
            return text, False
            
        def has_pii(self, text):
            return False
    
    def detect_language(text):
        return "en", 1.0
    
    def load_reproductive_health_data():
        return {}

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages the conversation flow.
    NOTE: This implementation seems older. In the multi-aspect architecture,
    MultiAspectQueryProcessor would typically be the main entry point.
    This version is updated for consistency in state/greeting detection,
    but may duplicate logic found elsewhere.
    """

    def __init__(self, evaluation_model="local"):
        """
        Initialize the conversation manager

        Args:
            evaluation_model (str): Model to use for response evaluation
        """
        logger.info(f"Initializing Conversation Manager with evaluation_model={evaluation_model}")
        try:
            # Initialize components - consider passing instances if managed centrally
            # Using internal instances for now
            self.baseline_model = BaselineModel(evaluation_model=evaluation_model) # Assuming this exists and is configured
            self.friendly_bot = FriendlyBot() # Assuming this exists
            self.citation_manager = CitationManager()
            self.policy_api = PolicyAPI() # Internal instance for state name mapping, etc.
            # self.question_classifier = QuestionClassifier() # Classification likely handled upstream
            self.context_manager = ContextManager(max_context_length=5) # Increased context slightly
            self.conversation_history = [] # Use ContextManager primarily now
            self._session_ended = False
            self.current_session_id = str(uuid.uuid4()) # Default session ID

            # Initialize PII detector
            self.pii_detector = PIIDetector()

            # Set up log file path
            self.log_dir = "logs"
            os.makedirs(self.log_dir, exist_ok=True)
            self.conversation_log_file = os.path.join(self.log_dir, f"conversation_log_{self.current_session_id}.json") # Log per session

            # Add policy response cache (can be helpful even here)
            self.policy_cache = {}
            self.cache_ttl = 3600 * 6 # 6 hours

            # Evaluation frequency
            self.evaluation_frequency = int(os.environ.get("EVALUATION_FREQUENCY", "10"))
            self.message_counter = 0

            # Define ambiguous state abbreviations (consistent with PolicyHandler)
            self.AMBIGUOUS_ABBREVIATIONS = {"in", "on", "at", "me", "hi", "ok", "or", "la", "pa", "no", "so", "de", "oh", "co", "wa", "va"}

            logger.info("Conversation Manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Conversation Manager: {str(e)}",
                         exc_info=True)
            raise

    def _save_conversation_logs(self):
        """Save conversation logs for the current session."""
        # This should ideally use the MemoryManager for persistence if integrated
        # Keeping basic file logging here for standalone functionality
        try:
            log_data = self.context_manager.get_context() # Get messages from context manager
            # Sanitize PII one last time before saving
            sanitized_log = []
            for msg in log_data:
                 entry = msg.copy()
                 if entry.get('sender') == 'user' and isinstance(entry.get('message'), str):
                      sanitized_message, _ = self.pii_detector.redact_pii(entry['message'])
                      entry['message'] = sanitized_message
                      entry.pop('original_message', None) # Don't save original PII
                 # Ensure timestamp is ISO format string
                 if isinstance(entry.get('timestamp'), float):
                      entry['timestamp'] = datetime.fromtimestamp(entry['timestamp']).isoformat()
                 sanitized_log.append(entry)

            # Ensure log directory exists
            os.makedirs(os.path.dirname(self.conversation_log_file), exist_ok=True)

            with open(self.conversation_log_file, 'w') as f:
                json.dump(sanitized_log, f, indent=2)
            logger.debug(f"Saved {len(sanitized_log)} messages to {self.conversation_log_file}")

        except Exception as e:
            logger.error(f"Error saving conversation logs for session {self.current_session_id}: {str(e)}")


    def _extract_zip_code(self, message):
        """Extract 5-digit zip code from message"""
        if not message or not isinstance(message, str):
            return None
        # Look for standard 5-digit US zip code using word boundaries
        zip_pattern = r'\b(\d{5})\b'
        match = re.search(zip_pattern, message)
        if match:
            zip_code = match.group(1)
            logger.info(f"Extracted zip code: {zip_code}")
            return zip_code
        return None

    def process_message(self, message: str, conversation_id: Optional[str] = None, persona: str = "baseline") -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        NOTE: This logic might be superseded by MultiAspectQueryProcessor in a full setup.
        """
        message_id = str(uuid.uuid4())

        # Set or update session ID
        if conversation_id and conversation_id != self.current_session_id:
            logger.info(f"Switching to conversation ID: {conversation_id}")
            self.current_session_id = conversation_id
            # Potentially load history for this ID using MemoryManager if integrated
            self.conversation_log_file = os.path.join(self.log_dir, f"conversation_log_{self.current_session_id}.json")
            self.message_counter = 0 # Reset counter for new session
            # self.context_manager.clear_context() # Clear context for new session? Or load existing? Depends on desired behavior.
        elif not conversation_id:
             # If no ID provided, stick with the current one or generate a new one if needed
             conversation_id = self.current_session_id
             logger.debug(f"Using current session ID: {self.current_session_id}")


        # Add user message to context/history (sanitized)
        sanitized_message, pii_warning = self._preprocess_input(message)
        self.add_to_history('user', sanitized_message, evaluate=False) # Store sanitized version

        # Handle PII warning if needed (should be handled by preprocessor ideally)
        if pii_warning:
             logger.warning("PII detected in input, returning warning.")
             response = {
                  "text": pii_warning,
                  "citations": [], "citation_objects": [], "message_id": message_id
             }
             self.add_to_history('bot', pii_warning, message_id=message_id)
             return response

        # --- FIX (Bug 4): Check for simple greetings/phrases ---
        message_lower = sanitized_message.lower().strip()
        simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hi there", "heya", "yo", "what's up", "sup"]
        simple_how_are_you = ["how are you", "how's it going", "how are things", "how you doing", "how are you doing"]
        simple_thanks = ["thanks", "thank you", "ty", "thx", "appreciate it"]
        simple_bye = ["bye", "goodbye", "later", "see ya", "farewell", "take care"]

        is_simple_greeting = (
            (message_lower in simple_greetings and len(sanitized_message.split()) <= 3) or
            (message_lower in simple_how_are_you and len(sanitized_message.split()) <= 5)
        )
        is_simple_thanks = (message_lower in simple_thanks and len(sanitized_message.split()) <= 3)
        is_simple_bye = (message_lower in simple_bye and len(sanitized_message.split()) <= 3)

        if is_simple_greeting:
             logger.info("Handling simple greeting in ConversationManager.")
             response_text = random.choice([
                  "Hi! I'm doing well, thank you for checking in. How are you feeling today? If you ever have questions about reproductive health, I'm here for you 💜",
                  "Hello there! Thanks for asking. I'm functioning well and ready to help with any reproductive health questions you have.",
                  "Hey! I'm here and ready to assist. I hope you're having a good day! What's on your mind regarding reproductive health?"
             ])
             response = {"text": response_text, "citations": [], "citation_objects": [], "message_id": message_id}
             self.add_to_history('bot', response_text, message_id=message_id)
             return response
        if is_simple_thanks:
             logger.info("Handling simple thanks in ConversationManager.")
             response_text = random.choice(["You're very welcome! 😊", "Happy to help!", "Anytime!"])
             response = {"text": response_text, "citations": [], "citation_objects": [], "message_id": message_id}
             self.add_to_history('bot', response_text, message_id=message_id)
             return response
        if is_simple_bye:
             logger.info("Handling simple bye in ConversationManager.")
             response_text = random.choice(["Take care!", "Goodbye! Feel free to reach out anytime.", "Wishing you well!"])
             response = {"text": response_text, "citations": [], "citation_objects": [], "message_id": message_id}
             self.add_to_history('bot', response_text, message_id=message_id)
             return response
        # --- End Bug 4 Fix ---

        # Check for policy context using refined state detection
        # Determine if the query is primarily about policy
        is_policy_q = self._is_policy_lookup(sanitized_message)
        state_code = self._check_for_state_names(sanitized_message)
        zip_code = self._extract_zip_code(sanitized_message)

        if not state_code and zip_code:
            state_code = self._get_state_from_zip_code(zip_code) # Use internal ZIP lookup

        # If it's a policy question AND we have a state, handle it
        if is_policy_q and state_code:
            logger.info(f"Handling policy question for state: {state_code}")
            policy_response = self._get_policy_response_for_state(state_code, message_id, zip_code)
            self.add_to_history('bot', policy_response['text'], message_id=message_id) # Log bot response
            return policy_response
        # If it's likely policy but no state found, ask for state
        elif is_policy_q and not state_code:
            logger.info("Policy question detected, but no state found. Asking for location.")
            response_text = "I can help with abortion policy information. Could you please tell me which U.S. state or ZIP code you're asking about?"
            response = {"text": response_text, "citations": [], "citation_objects": [], "message_id": message_id}
            self.add_to_history('bot', response_text, message_id=message_id)
            return response
        # Handle state-only messages (check after greetings)
        elif self.is_state_only_message(sanitized_message):
             logger.info("Handling state-only message.")
             state_only_code = self._check_for_state_names(sanitized_message) # Re-extract state code
             if state_only_code:
                  # Check history context to see if it's likely policy related
                  if self._has_recent_abortion_context():
                       logger.info(f"State-only message '{sanitized_message}' follows abortion context. Getting policy.")
                       policy_response = self._get_policy_response_for_state(state_only_code, message_id)
                       self.add_to_history('bot', policy_response['text'], message_id=message_id)
                       return policy_response
                  else:
                       logger.info(f"State-only message '{sanitized_message}' without clear abortion context. Asking for clarification.")
                       state_name_display = self.policy_api.STATE_NAMES.get(state_only_code, state_only_code)
                       response_text = f"I see you mentioned {state_name_display}. How can I help you regarding this state? Are you looking for information on reproductive health policies or services?"
                       response = {"text": response_text, "citations": [], "citation_objects": [], "message_id": message_id}
                       self.add_to_history('bot', response_text, message_id=message_id)
                       return response
             else:
                  # Should not happen if is_state_only_message was true, but handle defensively
                  logger.warning("is_state_only_message was true but _check_for_state_names failed.")
                  # Fall through to baseline model


        # If not a clear policy question or greeting, use the baseline model
        logger.info("Message not identified as policy or simple greeting. Using baseline model.")
        try:
            # Pass context from ContextManager to baseline model
            history_context = self.context_manager.get_context()
            response = self.baseline_model.process_message(sanitized_message, conversation_id, persona, history=history_context)

            # Add response to history and context manager
            if response and 'text' in response:
                self.add_to_history('bot', response['text'], message_id=response.get('message_id', message_id), evaluate=True) # Evaluate baseline responses

            return response
        except Exception as e:
            logger.error(f"Error in baseline processing: {str(e)}")
            error_response = {
                "text": "I apologize, but I encountered an error processing your message. Could you please try rephrasing your question?",
                "citations": [],
                "citation_objects": [],
                "message_id": message_id
            }
            self.add_to_history('bot', error_response['text'], message_id=message_id)
            return error_response
        finally:
             self._save_conversation_logs() # Save logs after each turn


    def _get_policy_response_for_state(self, state_code: str, message_id: str, zip_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get formatted policy response for a state using PolicyAPI data.
        This mirrors logic that might be in PolicyHandler or ResponseComposer.
        """
        try:
            # Check cache first
            cache_key = f"policy_{state_code}"
            cached_data = self.policy_cache.get(cache_key)
            if cached_data and time.time() - cached_data.get('timestamp', 0) < self.cache_ttl:
                 logger.info(f"Using cached policy response for {state_code}")
                 cached_data['data']['message_id'] = message_id # Update message ID
                 return cached_data['data']

            # Get raw policy data from API (or its cache)
            policy_data = self.policy_api.get_policy_data(state_code)
            state_name = self.policy_api.STATE_NAMES.get(state_code, state_code)

            if policy_data.get("error"):
                logger.warning(f"Policy API returned error/fallback for {state_code}")
                response_text = policy_data.get('message', f"I couldn't retrieve specific policy details for {state_name}. Please consult reliable sources like Planned Parenthood or AbortionFinder.org.")
                citations = policy_data.get('sources', []) # Use fallback sources
                citation_objects = citations # Assume fallback sources are already objects
            else:
                # Format the response (simplified formatting here)
                response_text = f"Here's a summary of abortion access in {state_name}:\n\n"
                details = self._extract_simple_policy_details(policy_data, state_code) # Basic extraction

                response_text += f"**Legality:** {details.get('Legality', 'Info unavailable')}\n"
                response_text += f"**Gestational Limit:** {details.get('Gestational Limit', 'Info unavailable')}\n"
                if 'Waiting Period' in details:
                    response_text += f"**Waiting Period:** {details['Waiting Period']}\n"
                if "Minors Access" in details:
                    response_text += f"**Minors Access:** {details['Minors Access']}\n"

                # Add clinic info if zip provided (basic placeholder)
                if zip_code:
                     response_text += f"\nTo find clinics near {zip_code}, please visit AbortionFinder.org."

                response_text += "\n\n*Note: Laws can change. Please verify with official sources.*"
                # Get citations (use default policy sources)
                citations_list = self._get_default_policy_citations(state_code)
                citations = [c['source'] for c in citations_list]
                citation_objects = citations_list

            # Construct response object
            response = {
                'text': response_text,
                'citations': citations,
                'citation_objects': citation_objects,
                'message_id': message_id,
                'state_code': state_code, # Include state code
                'state_name': state_name
            }

            # Cache the result (only if not an error)
            if not policy_data.get("error"):
                 self.policy_cache[cache_key] = {'timestamp': time.time(), 'data': response.copy()}

            return response

        except Exception as e:
            logger.error(f"Error getting policy response for state {state_code}: {str(e)}")
            state_name = self.policy_api.STATE_NAMES.get(state_code, "that state")
            fallback_citations = self._get_default_policy_citations(state_code)
            return {
                'text': f"I apologize, but I encountered an error while retrieving policy information for {state_name}. Please try again or check official resources like AbortionFinder.org.",
                'citations': [c['source'] for c in fallback_citations],
                'citation_objects': fallback_citations,
                'message_id': message_id
            }

    def _extract_simple_policy_details(self, policy_data: Dict[str, Any], state_code: str) -> Dict[str, str]:
         """Basic extraction of policy details from PolicyAPI structure."""
         details = {}
         endpoints = policy_data.get("endpoints", {})

         # Gestational Limits
         gestational = endpoints.get("gestational_limits", {}).get(state_code, {})
         if gestational:
              if gestational.get("banned"): details['Legality'] = "Banned with limited exceptions"
              elif gestational.get("banned_after_weeks_since_LMP") is not None:
                   weeks = gestational["banned_after_weeks_since_LMP"]
                   if weeks == 0: details['Legality'] = "Banned with limited exceptions"
                   elif weeks == 99: details['Gestational Limit'] = "Up to viability"
                   else: details['Gestational Limit'] = f"Up to {weeks} weeks LMP"
              else: details['Legality'] = "Generally Legal" # Default if not explicitly banned
         else: details['Legality'] = "Info unavailable"

         # Waiting Periods
         waiting = endpoints.get("waiting_periods", {}).get(state_code, {})
         if waiting and waiting.get("waiting_period_hours", 0) > 0:
              details['Waiting Period'] = f"{waiting['waiting_period_hours']} hours"

         # Minors
         minors = endpoints.get("minors", {}).get(state_code, {})
         if minors:
              if minors.get("parental_consent_required"): details['Minors Access'] = "Parental consent required"
              elif minors.get("parental_notification_required"): details['Minors Access'] = "Parental notification required"
              else: details['Minors Access'] = "No parental involvement required"

         return details

    def _get_default_policy_citations(self, state_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns default citation objects for policy questions."""
        state_name = self.policy_api.STATE_NAMES.get(state_code, "the relevant state") if state_code else "U.S. States"
        accessed_date = datetime.now().strftime('%Y-%m-%d')
        return [
            {
                "id": f"guttmacher-{state_code.lower() if state_code else 'general'}",
                "source": "Guttmacher Institute",
                "url": "https://www.guttmacher.org/state-policy/explore/abortion-policy",
                "title": f"Abortion Policy in {state_name}",
                "accessed_date": accessed_date
            },
            {
                 "id": f"finder-{state_code.lower() if state_code else 'general'}",
                 "source": "AbortionFinder.org",
                 "url": "https://www.abortionfinder.org/",
                 "title": "Find Verified Abortion Care",
                 "accessed_date": accessed_date
            }
        ]


    def _check_for_state_names(self, message: str) -> Optional[str]:
        """
        Check for state names in a message using updated logic.

        Args:
            message (str): Message to check for state names

        Returns:
            str: State code if found, None otherwise
        """
        # Reusing the refined logic from PolicyAPI's extraction method
        return self.policy_api._extract_state_from_question(message)


    def _is_zip_code(self, text: str) -> bool:
        """Check if a string is likely a valid US zip code."""
        if not text or not isinstance(text, str):
             return False
        # Check for 5 digits, optionally allowing surrounding non-digits
        match = re.search(r'\b(\d{5})\b', text)
        return bool(match)

    def _get_state_from_zip_code(self, zip_code: str) -> Optional[str]:
        """
        Convert a zip code to a state code using available methods.

        Args:
            zip_code (str): US ZIP code

        Returns:
            str: Two-letter state code or None if not found
        """
        # Extract 5 digits first if not already done
        zip_match = re.search(r'\b(\d{5})\b', str(zip_code))
        if not zip_match:
            logger.warning(f"Could not extract clean 5-digit ZIP from '{zip_code}' for lookup.")
            return None
        
        clean_zip = zip_match.group(1)
        
        # First try our new direct lookup function
        try:
            from utils.zip_codes import zip_code_to_state_code
            state_code = zip_code_to_state_code(clean_zip)
            if state_code:
                logger.info(f"Found state {state_code} for ZIP {clean_zip} using direct lookup")
                return state_code
        except ImportError:
            logger.warning("Could not import zip_code_to_state_code function, falling back to other methods")
        
        # Use preprocessor's method if available
        if hasattr(self.preprocessor, 'get_state_from_zip'):
            state = self.preprocessor.get_state_from_zip(clean_zip)
            if state: 
                logger.info(f"Found state {state} for ZIP {clean_zip} using preprocessor")
                return state

        # Fallback using PolicyAPI's internal method (which includes pyzipcode check and range fallback)
        if hasattr(self.policy_api, '_get_state_from_zip_fallback'):
            state = self.policy_api._get_state_from_zip_fallback(clean_zip)
            if state:
                logger.info(f"Found state {state} for ZIP {clean_zip} using policy API fallback")
                return state
        else:
            logger.error("PolicyAPI fallback ZIP method not found.")
        
        logger.warning(f"Could not determine state for ZIP code '{clean_zip}' using any method")
        return None


    def _is_abortion_related(self, message: str) -> bool:
        """Check if a message is related to abortion access or policy"""
        if not message or not isinstance(message, str):
            return False

        message_lower = message.lower()

        # Define patterns that indicate abortion-related context
        abortion_terms = [
            'abortion', 'terminate pregnancy', 'pregnancy termination',
            'end pregnancy', 'end a pregnancy', 'end my pregnancy',
            'abortion pill', 'abortion clinic', 'abortion provider',
            'roe v wade', 'legal abortion', 'abortion access', 'abortion restriction',
            'abortion law', 'abortion policy', 'abortion service', 'abortion right',
            'abortion ban', 'abortion legal', 'abortion illegal', 'gestational limit',
            'weeks pregnant abortion' # More specific phrases
        ]

        # Check for phrases asking about abortion access
        abortion_access_patterns = [
            'can i get an abortion', 'i need an abortion', 'where can i get an abortion',
            'how to get an abortion', 'abortion near me', 'abortion clinic near',
            'abortion provider near', 'find abortion', 'need abortion', 'want abortion',
            'where to get abortion', 'how to access abortion', 'looking for abortion',
            'is abortion legal', 'is abortion available', 'is abortion allowed',
            'abortion options', 'abortion services', 'getting an abortion',
            'seeking abortion', 'access to abortion'
        ]

        # Check for generic terms in message
        if any(term in message_lower for term in abortion_terms):
            return True

        # Check for access patterns
        if any(pattern in message_lower for pattern in abortion_access_patterns):
            return True

        # Check for state policy questions explicitly mentioning abortion
        policy_keywords = ['state', 'legal', 'allowed', 'law', 'policy', 'regulation', 'ban', 'restriction']
        if 'abortion' in message_lower and any(word in message_lower for word in policy_keywords):
            return True

        return False


    def _has_recent_abortion_context(self) -> bool:
        """Check if there's abortion-related context in recent messages (from ContextManager)"""
        history = self.context_manager.get_context()
        if not history:
            return False

        # Check last 3 messages for abortion context
        for entry in reversed(history[-3:]):
            message_text = entry.get('message', '')
            if message_text and isinstance(message_text, str):
                 if self._is_abortion_related(message_text):
                      logger.info("Found abortion context in recent history via ContextManager")
                      return True
                 # Check if bot asked for state in relation to abortion
                 if entry.get('sender') == 'bot' and ('abortion' in message_text.lower() or 'policy' in message_text.lower()) and ('state' in message_text.lower() or 'zip code' in message_text.lower()):
                      logger.info("Found bot prompting for location regarding abortion in recent history.")
                      return True

        return False

    def _preprocess_input(self, message: str) -> Tuple[str, Optional[str]]:
        """Basic preprocessing: cleaning and PII check."""
        # 1. Clean message
        message = self.clean_message(message)

        # 2. PII Check
        sanitized_message = message
        warning = None
        if self.pii_detector.has_pii(message):
             sanitized_message, _ = self.pii_detector.redact_pii(message) # Use updated redaction
             warning = (
                 "I noticed some personal information (like name, phone, or email) in your message. "
                 "For your privacy, I've removed it. Please avoid sharing sensitive personal details. "
                 "How can I help with your reproductive health questions?"
             )
             logger.info(f"PII sanitized. Original length: {len(message)}, Sanitized length: {len(sanitized_message)}")

        # 3. Language Check (Optional - Assuming handled by Preprocessor if integrated)
        # language = detect_language(sanitized_message)
        # if language != 'en':
        #     warning = "I currently only support English. Please ask your question in English."
        #     sanitized_message = "[Non-English message detected]" # Replace content

        return sanitized_message, warning


    def add_to_history(self, sender: str, message: str, message_id: Optional[str] = None, evaluate: bool = False):
        """
        Add a message to the conversation history via ContextManager.

        Args:
            sender (str): The sender ('user' or 'bot')
            message (str): The message content
            message_id (str, optional): Unique ID for the message
            evaluate (bool): Whether to trigger evaluation for bot messages
        """
        if not message: return # Don't add empty messages

        message_id = message_id or str(uuid.uuid4())
        timestamp = time.time() # Use timestamp float for context manager

        # Prepare message entry for ContextManager
        message_entry = {
            'session_id': self.current_session_id,
            'message_id': message_id,
            'sender': sender,
            'message': message, # Store the potentially sanitized message
            'timestamp': timestamp,
            'type': 'message' # Context manager might use type
        }

        # Add evaluation data if applicable (simplified example)
        if sender == 'bot' and evaluate:
             self.message_counter += 1
             if self.message_counter % self.evaluation_frequency == 0:
                  logger.info(f"Evaluating bot message {message_id} (message count: {self.message_counter})")
                  # Placeholder for actual evaluation call
                  # evaluation_data = self.baseline_model.response_evaluator.evaluate(...)
                  # message_entry['evaluation'] = evaluation_data
                  pass # Add evaluation logic here if needed


        # Add to ContextManager
        self.context_manager.add_message(message_entry)
        logger.debug(f"Added message {message_id} ({sender}) to context manager.")

        # Note: Saving logs is handled after the full turn in process_message


    def get_history(self) -> List[Dict[str, Any]]:
        """Get the recent conversation history from the ContextManager."""
        # If session ended flag is set, maybe return empty or signal UI differently?
        # if self._session_ended:
        #     logger.info("Session marked as ended, returning empty history for UI (if desired)")
        #     return []
        return self.context_manager.get_context()

    def clear_history(self):
        """Clears the context managed by ContextManager for the current session."""
        try:
            logger.info(f"Clearing context history for session {self.current_session_id}")
            # Save logs before clearing context if needed (depends on workflow)
            # self._save_conversation_logs() # Save final state before clearing?
            self.context_manager.clear_context()
            self._session_ended = True # Mark session as ended
            # Optionally reset message counter
            self.message_counter = 0
            # Generate a new session ID for the next interaction
            self.current_session_id = str(uuid.uuid4())
            self.conversation_log_file = os.path.join(self.log_dir, f"conversation_log_{self.current_session_id}.json")
            logger.info(f"History cleared. New session ID: {self.current_session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return False


    def detect_location_context(self, message: str) -> Optional[str]:
        """Detects location context (state name/code) in a message."""
        # Use the refined state checking method
        state_code = self._check_for_state_names(message)
        if state_code:
             # Return the state code directly
             return state_code
        # Optionally check for ZIP and convert if needed, but _check_for_state_names is primary
        # zip_code = self._extract_zip_code(message)
        # if zip_code:
        #     return self._get_state_from_zip_code(zip_code)
        return None


    def clean_message(self, message: str) -> str:
        """Clean and basic preprocess a message"""
        if not message or not isinstance(message, str):
            return ""

        # Remove extra whitespace
        message = " ".join(message.split())

        # Optional: Remove special characters but keep basic punctuation and common symbols
        # This is a basic example, might need refinement based on expected input
        # message = re.sub(r'[^\w\s.,!?-]', '', message)

        return message.strip()

    def is_state_only_message(self, content: str) -> bool:
        """Check if the message primarily consists of only a state name or abbreviation."""
        if not content:
            return False

        cleaned_content = content.strip().lower()
        # Remove punctuation for matching
        cleaned_content_nopunct = re.sub(r'[^\w\s]', '', cleaned_content)

        # Check if the cleaned content exactly matches a state name or abbreviation
        state_names_lower = {name.lower() for name in self.policy_api.STATE_NAMES.values() if name}
        state_abbrevs_lower = {abbr.lower() for abbr in self.policy_api.STATE_NAMES.keys()}

        # Check exact match first
        if cleaned_content_nopunct in state_names_lower or cleaned_content_nopunct in state_abbrevs_lower:
             # Check if the original message was very short (e.g., <= 3 words)
             if len(content.split()) <= 3:
                  logger.info(f"Detected potential state-only message: {content}")
                  return True

        return False


    def handle_message(self, content: str) -> Dict[str, Any]:
        """Alias for process_message for potential external interface consistency."""
        # This method now seems redundant with process_message.
        # Keeping it as an alias for compatibility, but recommend using process_message directly.
        logger.warning("handle_message is deprecated, please use process_message.")
        return self.process_message(message=content)


    def _is_policy_lookup(self, message: str) -> bool:
        """Determine if a message is asking about abortion policy information."""
        # Keywords related to policy questions (expanded for better recall)
        policy_keywords = [
            'policy', 'policies', 'legal', 'illegal', 'allow', 'allowed', 'law',
            'laws', 'restrict', 'restriction', 'restrictions', 'state', 'states',
            'access', 'can i get', 'where can i', 'option', 'options', 'ban', 'banned',
            'regulation', 'require', 'mandate', 'coverage', 'insurance', 'medicaid',
            'gestational limit', 'waiting period', 'parental consent', 'parental notification'
        ]

        # Check for abortion context first
        is_abortion_context = self._is_abortion_related(message)

        # Check for policy keywords
        message_lower = message.lower()
        has_policy_keywords = any(keyword in message_lower for keyword in policy_keywords)

        # Require both abortion context and policy keywords OR a strong abortion access phrase
        strong_access_phrases = ['can i get an abortion', 'is abortion legal in', 'abortion access in']
        has_strong_access_phrase = any(phrase in message_lower for phrase in strong_access_phrases)

        is_policy = (is_abortion_context and has_policy_keywords) or has_strong_access_phrase

        if is_policy:
             logger.debug(f"Message classified as policy lookup: {message[:60]}...")
        return is_policy


    # Removed methods that seemed deprecated or duplicated elsewhere:
    # _format_response, _process_policy_question, _check_simple_queries,
    # _extract_location, _get_policy_response, _format_policy_response,
    # _handle_emotional_query, _handle_multi_faceted_query,
    # _is_emergency_contraception_query, _get_emergency_contraception_response,
    # _get_baseline_response

    # Kept helper methods used by the revised process_message:
    # _save_conversation_logs, _extract_zip_code, _get_policy_response_for_state,
    # _extract_simple_policy_details, _get_default_policy_citations, _check_for_state_names,
    # _is_zip_code, _get_state_from_zip_code, _is_abortion_related, _has_recent_abortion_context,
    # _preprocess_input, add_to_history, get_history, clear_history, detect_location_context,
    # clean_message, is_state_only_message, _is_policy_lookup


# --- END OF FILE conversation_manager.py ---