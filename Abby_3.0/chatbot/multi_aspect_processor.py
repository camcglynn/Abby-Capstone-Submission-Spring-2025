# --- START OF FILE multi_aspect_processor.py ---

import logging
import asyncio
import re
from typing import Dict, List, Any, Optional
import os
import time
import uuid
import random

from .unified_classifier import UnifiedClassifier
from .aspect_decomposer import AspectDecomposer
from .response_composer import ResponseComposer
from .knowledge_handler import KnowledgeHandler
from .emotional_support_handler import EmotionalSupportHandler
from .policy_handler import PolicyHandler
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)

class MultiAspectQueryProcessor:
    """
    Main orchestrator for the multi-aspect query handling process.

    This class manages the full lifecycle of a user query:
    1. Preprocessing (input validation and cleaning)
    2. Classification
    3. Aspect decomposition
    4. Specialized handling per aspect
    5. Response composition

    It maintains references to all specialized handlers and manages their execution.
    """

    def __init__(self,
                api_key: Optional[str] = None,
                openai_model: str = "gpt-4o-mini",
                policy_api_base_url: Optional[str] = None):
        """
        Initialize the multi-aspect query processor

        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            openai_model (str): OpenAI model to use
            policy_api_base_url (Optional[str]): Base URL for the abortion policy API
        """
        logger.info("Initializing MultiAspectQueryProcessor")

        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Allow initialization without API key for potential non-OpenAI flows
            logger.warning("OpenAI API key not found. Some features (classification, decomposition, OpenAI generation) might be unavailable.")
            # Raise error only if strictly required? For now, allow fallback.
            # raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        # Use provided policy API URL or get from environment
        self.policy_api_base_url = policy_api_base_url or os.getenv("POLICY_API_BASE_URL",
                                                                   "https://api.abortionpolicyapi.com/v1/")

        # Initialize preprocessing component
        self.preprocessor = Preprocessor()

        # Initialize components (handle potential missing API key)
        self.unified_classifier = None
        self.aspect_decomposer = None
        if self.api_key:
            try:
                self.unified_classifier = UnifiedClassifier(api_key=self.api_key, model_name=openai_model)
                self.aspect_decomposer = AspectDecomposer(api_key=self.api_key, model_name=openai_model)
                logger.info("Initialized OpenAI-dependent components: Classifier, Decomposer.")
            except Exception as e: # Catch potential initialization errors (like missing package)
                logger.error(f"Failed to initialize OpenAI dependent components: {e}")
                logger.warning("Running without UnifiedClassifier and AspectDecomposer. Will use basic routing.")
        else:
            logger.warning("Running without UnifiedClassifier and AspectDecomposer due to missing API key.")


        self.response_composer = ResponseComposer()

        # Initialize specialized handlers (handle potential missing API key)
        self.handlers = {}
        try:
             self.handlers["knowledge"] = KnowledgeHandler(api_key=self.api_key, model_name=openai_model)
        except Exception as e: logger.error(f"Failed to initialize KnowledgeHandler: {e}")
        try:
             self.handlers["emotional"] = EmotionalSupportHandler(api_key=self.api_key, model_name=openai_model)
        except Exception as e: logger.error(f"Failed to initialize EmotionalSupportHandler: {e}")
        try:
             self.handlers["policy"] = PolicyHandler(api_key=self.api_key, policy_api_base_url=self.policy_api_base_url)
        except Exception as e: logger.error(f"Failed to initialize PolicyHandler: {e}")

        # Performance tracking
        self.processing_times = {
            "preprocessing": [],
            "classification": [],
            "decomposition": [],
            "handling": {},
            "composition": []
        }
        # Initialize tracking for potentially available handlers
        for handler_type in ["knowledge", "emotional", "policy"]:
            self.processing_times["handling"][handler_type] = []

    async def process_query(self,
                          message: str,
                          conversation_history: List[Dict[str, Any]] = None,
                          user_location: Optional[Dict[str, str]] = None,
                          session_id: Optional[str] = None,
                          aspect_type: Optional[str] = None,
                          original_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query by decomposing it and handling each aspect separately.
        
        Args:
            message (str): User query to process
            conversation_history (List[Dict[str, Any]], optional): Previous conversation messages
            user_location (Optional[Dict[str, str]], optional): User's location data
            session_id (Optional[str], optional): Session identifier
            aspect_type (Optional[str], optional): Force specific aspect type (for testing)
            original_message (Optional[str], optional): Original unmodified user message
            
        Returns:
            Dict[str, Any]: Final composed response
        """
        # Store the start time for performance measurement
        start_time = time.time()
        
        # Save a copy of the original user message if not explicitly provided
        actual_user_input = original_message or message
        logger.debug(f"Processing query with original_message: '{actual_user_input}'")
        
        # Return early if we get an empty message to avoid unnecessary computation
        if not message or not message.strip():
            logger.warning("Received empty message, returning basic response.")
            return {
                "text": "I'm not sure what you're asking. Can you provide more details?",
                "primary_content": "I'm not sure what you're asking. Can you provide more details?",
                "session_id": session_id,
                "query": message,
                "processing_time": 0
            }

        session_id = session_id or str(uuid.uuid4()) # Ensure session_id exists
        # Add session ID to the response composer if needed
        if hasattr(self.response_composer, 'current_session_id'):
             self.response_composer.current_session_id = session_id

        # Store the actual user input for logging/tracking
        actual_user_input = original_message or message
        logger.info(f"Processing query for session {session_id}: {message[:100]}...")

        # --- FIX (Bug 4): Early check for simple greetings/phrases ---
        message_lower = message.lower().strip()
        # More robust lists, checking length as well
        simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hi there", "heya", "yo", "what's up", "sup"]
        simple_how_are_you = ["how are you", "how's it going", "how are things", "how you doing", "how are you doing"]
        simple_thanks = ["thanks", "thank you", "ty", "thx", "appreciate it"]
        simple_bye = ["bye", "goodbye", "later", "see ya", "farewell", "take care"]

        word_count = len(message.split())

        is_simple_greeting = (
            any(message_lower.startswith(g) for g in simple_greetings) and word_count <= 3 or
            message_lower in simple_how_are_you and word_count <= 5
        )
        is_simple_thanks = (message_lower in simple_thanks and word_count <= 3)
        is_simple_bye = (message_lower in simple_bye and word_count <= 3)

        # Handle Greetings
        if is_simple_greeting:
            logger.info("Detected simple greeting, bypassing complex processing.")
            greeting_responses = [
                "Hi! I'm doing well, thank you for checking in. How are you feeling today? If you ever have questions about reproductive health, I'm here for you 💜",
                "Hello there! Thanks for asking. I'm functioning well and ready to help with any reproductive health questions you have.",
                "Hey! I'm here and ready to assist. I hope you're having a good day! What's on your mind regarding reproductive health?"
            ]
            response_text = random.choice(greeting_responses)
            # Use ResponseComposer to format consistently
            response = await self.response_composer.compose_response(
                 message=message, # Pass original message for context if needed
                 aspect_responses=[{
                      "aspect_type": "emotional", # Treat as emotional/conversational
                      "text": response_text,
                      "primary_content": response_text,
                      "confidence": 0.95,
                      "citations": [], # Ensure empty citations for simple responses
                      "citation_objects": []
                 }],
                 classification={"primary_type": "greeting"} # Pass classification hint
            )
            response["session_id"] = session_id # Ensure session ID is included
            response["processing_time"] = time.time() - start_time # Add processing time
            response["message_id"] = str(uuid.uuid4()) # Generate a message ID
            response["timestamp"] = time.time() # Add timestamp
            if "graphics" not in response: response["graphics"] = [] # Ensure graphics key
            return response

        # Handle simple Thanks/Bye - provide minimal acknowledgement
        if is_simple_thanks:
             logger.info("Detected simple thanks.")
             response_text = random.choice(["You're very welcome! 😊", "Happy to help!", "Anytime!"])
             response = await self.response_composer.compose_response(
                 message=message,
                 aspect_responses=[{"aspect_type": "emotional", "text": response_text, "primary_content": response_text, "confidence": 0.9, "citations": [], "citation_objects": []}],
                 classification={"primary_type": "closing"}
             )
             response["session_id"] = session_id
             response["processing_time"] = time.time() - start_time
             response["message_id"] = str(uuid.uuid4())
             response["timestamp"] = time.time()
             if "graphics" not in response: response["graphics"] = []
             return response

        if is_simple_bye:
             logger.info("Detected simple goodbye.")
             response_text = random.choice(["Take care!", "Goodbye! Feel free to reach out anytime.", "Wishing you well!"])
             response = await self.response_composer.compose_response(
                 message=message,
                 aspect_responses=[{"aspect_type": "emotional", "text": response_text, "primary_content": response_text, "confidence": 0.9, "citations": [], "citation_objects": []}],
                 classification={"primary_type": "closing"}
             )
             response["session_id"] = session_id
             response["processing_time"] = time.time() - start_time
             response["message_id"] = str(uuid.uuid4())
             response["timestamp"] = time.time()
             if "graphics" not in response: response["graphics"] = []
             return response

        # --- End Bug 4 Fix ---


        try:
            # Initialize conversation history if None
            conversation_history = conversation_history or []

            # 0. Preprocess the query
            preprocessing_start = time.time()
            preprocess_result = self.preprocessor.process(message)
            preproc_metadata = preprocess_result.get("metadata", {}) # Safely get metadata
            self.processing_times["preprocessing"].append(time.time() - preprocessing_start)

            logger.info(f"Preprocessing result: {preproc_metadata.get('preprocessing', {})}")

            # If message is not processable (e.g., non-English), return early using composer
            if not preprocess_result.get('is_processable', True):
                logger.info(f"Message not processable: {preprocess_result.get('stop_reason')}")
                fallback_response = self.response_composer._create_fallback_response(
                    error_type=preprocess_result.get('stop_reason', 'general'),
                    context={"original_message": message}
                )
                fallback_response["session_id"] = session_id
                fallback_response["processing_time"] = time.time() - start_time
                return fallback_response


            # Use the processed message for further processing
            processed_message = preprocess_result.get('processed_message', message) # Fallback to original if missing

            # 1. Classify the query (if classifier is available)
            classification = None
            if self.unified_classifier:
                 classification_start = time.time()
                 try:
                     classification = await self.unified_classifier.classify(
                          processed_message,
                          conversation_history
                     )
                     self.processing_times["classification"].append(time.time() - classification_start)
                     logger.info(f"Unified classification result: {classification}")
                 except Exception as classify_err:
                      logger.error(f"Unified classifier failed: {classify_err}", exc_info=True)
                      classification = self._basic_classify_fallback(processed_message) # Use fallback
                      logger.warning(f"Using basic fallback classification due to error: {classification}")

            else:
                 # Basic fallback classification if no classifier
                 classification = self._basic_classify_fallback(processed_message)
                 logger.info(f"Basic fallback classification (no classifier): {classification}")

            # Add session_id to classification for potential use by handlers
            classification["session_id"] = session_id

            # 2. Decompose into aspects (if decomposer is available and needed)
            aspects = []
            is_multi_aspect = classification.get("is_multi_aspect", False)
            # Also consider multi-aspect if multiple confidence scores are high
            confidence_scores = classification.get("confidence_scores", {})
            high_conf_aspects = [aspect for aspect, score in confidence_scores.items() if score > 0.6]
            if len(high_conf_aspects) > 1:
                 is_multi_aspect = True

            if self.aspect_decomposer and is_multi_aspect:
                 decomposition_start = time.time()
                 try:
                      aspects = await self.aspect_decomposer.decompose(
                           processed_message,
                           classification,
                           conversation_history
                      )
                      self.processing_times["decomposition"].append(time.time() - decomposition_start)
                      logger.info(f"Decomposed into {len(aspects)} aspects")
                 except Exception as decomp_err:
                      logger.error(f"Aspect decomposer failed: {decomp_err}", exc_info=True)
                      # Fallback to single aspect based on primary type if decomposition fails
                      primary_type = classification.get("primary_type", "knowledge")
                      aspects = self._create_single_aspect(processed_message, classification, primary_type)
                      logger.warning("Using single aspect fallback due to decomposition error.")

            elif classification:
                 # If not multi-aspect or decomposer unavailable, create single aspect based on primary type
                 primary_type = classification.get("primary_type", "knowledge")
                 aspects = self._create_single_aspect(processed_message, classification, primary_type)
                 logger.info(f"Using single aspect based on primary classification: {primary_type}")
            else:
                 # Absolute fallback if classification failed entirely
                 aspects = [{"type": "knowledge", "query": processed_message, "confidence": 0.5, "topics": ["general"]}]
                 logger.warning("Classification failed entirely, defaulting to single knowledge aspect.")


            # 3. Process each aspect with specialized handlers
            aspect_responses = []
            aspect_tasks = []

            # Filter aspects to only those with available handlers
            valid_aspects = [a for a in aspects if a.get("type") in self.handlers]
            if len(valid_aspects) < len(aspects):
                 logger.warning(f"Skipping {len(aspects) - len(valid_aspects)} aspects due to missing handlers.")


            for aspect in valid_aspects:
                aspect_type = aspect.get("type")
                handler = self.handlers[aspect_type]
                # Determine query for handler: specific from decompose, else use effective query
                handler_query = aspect.get("query", processed_message)
                logger.debug(f"Using query for {aspect_type} handler: {handler_query[:100]}...")

                task = asyncio.create_task(
                    self._process_aspect(
                        handler=handler,
                        aspect=aspect,
                        message=handler_query, # Pass specific handler query
                        conversation_history=conversation_history,
                        user_location=user_location,
                        aspect_type=aspect_type,
                        original_message=actual_user_input, # Pass actual user input 
                        full_message_context=processed_message # Pass full message for better context
                    )
                )
                aspect_tasks.append(task)

            # Wait for all aspect processing to complete
            if aspect_tasks:
                handler_results = await asyncio.gather(*aspect_tasks)
                # Filter out None responses or errors before passing to composer
                aspect_responses = [r for r in handler_results if r and isinstance(r, dict) and not r.get("error")]


            # 4. Compose the final response
            composition_start = time.time()
            final_response = await self.response_composer.compose_response(
                message=processed_message, # Pass processed message to composer
                aspect_responses=aspect_responses,
                classification=classification
            )
            self.processing_times["composition"].append(time.time() - composition_start)

            # Add common metadata
            final_response["session_id"] = session_id
            final_response["message_id"] = final_response.get("message_id", str(uuid.uuid4())) # Ensure message ID exists
            final_response["timestamp"] = final_response.get("timestamp", time.time())
            final_response["processing_time"] = time.time() - start_time
            final_response["preprocessing_metadata"] = final_response.get("preprocessing_metadata", {})
            final_response["preprocessing_metadata"]["input"] = preproc_metadata # Add preprocessor info

            # Add original user input to metadata for clarity
            final_response.setdefault("preprocessing_metadata", {}).setdefault("input", {})["original_input"] = actual_user_input

            # Add graphics key if not present
            if "graphics" not in final_response:
                 final_response["graphics"] = []


            return final_response

        except Exception as e:
            logger.error(f"Critical error in process_query: {str(e)}", exc_info=True)
            # Use composer's fallback for consistent error message formatting
            fallback_resp = self.response_composer._create_fallback_response(
                 error_type="technical",
                 context={"error": str(e), "stage": "main_processor"}
            )
            fallback_resp["session_id"] = session_id
            fallback_resp["processing_time"] = time.time() - start_time
            return fallback_resp

    def _create_single_aspect(self, message: str, classification: Dict[str, Any], primary_type: str) -> List[Dict[str, Any]]:
         """Helper to create a single aspect list based on classification."""
         return [{
             "type": primary_type,
             "query": message, # Use the full message for the single aspect
             "confidence": classification.get("confidence_scores", {}).get(primary_type, 0.8),
             "topics": classification.get("topics", ["reproductive_health"]),
             # Determine if state context is needed based on classification/type
             "requires_state_context": primary_type == "policy" and classification.get("contains_location", False)
         }]


    def _basic_classify_fallback(self, message: str) -> Dict[str, Any]:
         """Basic rule-based classification if UnifiedClassifier fails or is unavailable."""
         message_lower = message.lower()
         # Default classification
         classification = {
             "primary_type": "knowledge",
             "is_multi_aspect": False,
             "confidence_scores": {"knowledge": 0.6, "emotional": 0.2, "policy": 0.2},
             "topics": ["general", "reproductive_health"],
             "sensitive_content": [],
             "contains_location": False,
             "detected_locations": [],
             "query_complexity": "simple"
         }

         # Simple checks
         policy_keywords = ["policy", "law", "legal", "ban", "access", "state", "texas", "california", "florida", "new york"] # Expanded examples
         emotional_keywords = ["feel", "scared", "anxious", "worried", "sad", "depressed", "help me", "confused", "stress"]

         contains_policy = any(term in message_lower for term in policy_keywords)
         contains_emotional = any(term in message_lower for term in emotional_keywords)

         # Determine primary type based on keywords
         if contains_policy:
             classification["primary_type"] = "policy"
             classification["confidence_scores"] = {"knowledge": 0.2, "emotional": 0.1, "policy": 0.7}
             # Basic location check
             classification["contains_location"] = any(state in message_lower for state in ["texas", "california", "florida", "new york"]) or re.search(r'\b\d{5}\b', message) is not None
         elif contains_emotional:
             classification["primary_type"] = "emotional"
             classification["confidence_scores"] = {"knowledge": 0.2, "emotional": 0.7, "policy": 0.1}

         # Crude multi-aspect check
         if contains_policy and contains_emotional:
             classification["is_multi_aspect"] = True
             # Adjust confidences for multi-aspect
             classification["confidence_scores"] = {"knowledge": 0.1, "emotional": 0.5, "policy": 0.5}
         elif (contains_policy or contains_emotional) and classification["primary_type"] != "knowledge":
             # Check if knowledge keywords are also present
             knowledge_keywords = ["what is", "how does", "explain", "information", "types of", "side effects"]
             if any(term in message_lower for term in knowledge_keywords):
                  classification["is_multi_aspect"] = True
                  # Adjust confidences


         return classification


    async def _process_aspect(self,
                             handler: Any,
                             aspect: Dict[str, Any],
                             message: str,
                             conversation_history: List[Dict[str, Any]],
                             user_location: Optional[Dict[str, str]],
                             aspect_type: str,
                             original_message: str = None,
                             full_message_context: str = None) -> Optional[Dict[str, Any]]:
        """
        Process a single aspect using the appropriate handler, adding error handling.

        Args:
            handler: The specialized handler instance.
            aspect: The aspect data from decomposition.
            message: Processed user message.
            conversation_history: Previous conversation messages.
            user_location: User's location data.
            aspect_type: Type of the aspect (e.g., "knowledge", "policy").
            original_message: Original user message before preprocessing or modification.
            full_message_context: Full message context for broader context understanding.

        Returns:
            Optional[Dict[str, Any]]: Handler response dictionary or None if failed/empty.
        """
        response = None # Initialize response
        try:
            start_time = time.time()

            # Extract the aspect query, fall back to the main processed message
            aspect_query = aspect.get("query", message)
            confidence = aspect.get("confidence", 0.8)

            # Prepare arguments for the handler's process_query method
            handler_args = {
                 "query": aspect_query,
                 "full_message": full_message_context, # Pass the full message context
                 "conversation_history": conversation_history,
                 "user_location": user_location
            }
            
            # Add original_message only for handlers that accept it (like PolicyHandler)
            if aspect_type == "policy" and original_message:
                handler_args["original_message"] = original_message

            # Call the handler's process_query method
            # Assumes all handlers have an async process_query method
            # Check if handler is awaitable (async) or not
            if asyncio.iscoroutinefunction(handler.process_query):
                 response = await handler.process_query(**handler_args)
            else:
                 # If handler is synchronous, run it in default executor
                 # loop = asyncio.get_event_loop()
                 # response = await loop.run_in_executor(None, handler.process_query, handler_args)
                 # Simpler approach if sync handlers are okay: just call directly (might block loop)
                 logger.warning(f"Handler for {aspect_type} is synchronous. Calling directly.")
                 response = handler.process_query(**handler_args)


            # Validate and enrich the response
            if response and isinstance(response, dict):
                 # Ensure essential keys are present
                 response["aspect_type"] = response.get("aspect_type", aspect_type) # Use handler's type if provided
                 response["confidence"] = response.get("confidence", confidence) # Use handler's confidence if provided
                 # Ensure primary_content exists, use 'text' as fallback
                 response["primary_content"] = response.get("primary_content") or response.get("text", "")


                 # Only return if there's actual content to display
                 if response["primary_content"]:
                      # Track processing time
                      elapsed_time = time.time() - start_time
                      # Ensure handler_type exists in tracking dict
                      if aspect_type not in self.processing_times["handling"]:
                           self.processing_times["handling"][aspect_type] = []
                      self.processing_times["handling"][aspect_type].append(elapsed_time)
                      logger.info(f"Processed '{aspect_type}' aspect in {elapsed_time:.2f}s")
                      return response
                 else:
                      logger.warning(f"Handler for {aspect_type} returned response with empty content.")
                      return None # Return None if response is empty
            else:
                logger.warning(f"Handler for {aspect_type} returned invalid response: {response}")
                return None # Return None for invalid responses

        except Exception as e:
            logger.error(f"Error processing aspect '{aspect_type}' with handler {type(handler).__name__}: {str(e)}", exc_info=True)
            # Return a structured error response if processing fails
            return {
                "aspect_type": aspect_type,
                "primary_content": f"Sorry, I encountered an issue while handling the '{aspect_type}' part of your request.",
                "text": f"Sorry, I encountered an issue while handling the '{aspect_type}' part of your request.",
                "error": True,
                "error_message": str(e),
                "confidence": 0.0,
                "citations": [],
                "citation_objects": []
            }


    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the query processor

        Returns:
            Dict[str, Any]: Performance metrics including average processing times.
        """
        metrics = {
            "average_times": {},
            "total_queries_processed": 0, # Will be updated
            "handler_usage": {}
        }

        # Helper to calculate average safely
        def calculate_average(times_list):
            return sum(times_list) / len(times_list) if times_list else 0

        # Calculate average times for each step
        metrics["average_times"]["preprocessing"] = calculate_average(self.processing_times["preprocessing"])
        metrics["average_times"]["classification"] = calculate_average(self.processing_times["classification"])
        metrics["average_times"]["decomposition"] = calculate_average(self.processing_times["decomposition"])
        metrics["average_times"]["composition"] = calculate_average(self.processing_times["composition"])

        # Calculate average times and usage count for each handler
        total_handler_time = 0
        total_handler_calls = 0
        for handler_type, times in self.processing_times["handling"].items():
            avg_time = calculate_average(times)
            count = len(times)
            metrics["average_times"][f"handling_{handler_type}"] = avg_time
            metrics["handler_usage"][handler_type] = count
            total_handler_time += sum(times)
            total_handler_calls += count

        # Calculate overall average handling time
        metrics["average_times"]["handling_overall"] = total_handler_time / total_handler_calls if total_handler_calls else 0

        # Total queries processed (use classification count as a proxy, fallback to others)
        metrics["total_queries_processed"] = len(self.processing_times["classification"]) or len(self.processing_times["preprocessing"])


        # Calculate total average processing time per query
        num_queries = metrics["total_queries_processed"]
        if num_queries > 0:
             total_avg_time = (sum(self.processing_times["preprocessing"]) +
                               sum(self.processing_times["classification"]) +
                               sum(self.processing_times["decomposition"]) +
                               total_handler_time + # Sum of all handler times
                               sum(self.processing_times["composition"])) / num_queries
             metrics["average_times"]["total_processing_per_query"] = total_avg_time
        else:
             metrics["average_times"]["total_processing_per_query"] = 0


        return metrics

# --- END OF FILE multi_aspect_processor.py ---