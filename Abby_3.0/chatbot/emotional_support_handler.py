# --- START OF FILE emotional_support_handler.py ---

import logging
import os
import time
import re
import random
from typing import Dict, List, Any, Optional
from datetime import datetime # Added for citation date
# Use AsyncOpenAI if available for async processing
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    from openai import OpenAI # Fallback to sync OpenAI

logger = logging.getLogger(__name__)

class EmotionalSupportHandler:
    """
    Handler for emotional support aspects of user queries.

    This class processes messages seeking emotional support,
    validating user emotions and providing compassionate responses.
    Uses OpenAI for response generation.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the emotional support handler

        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use
        """
        logger.info(f"Initializing EmotionalSupportHandler with model {model_name}")

        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
             # Allow initialization without API key, but log warning
             logger.warning("OpenAI API key not found. Emotional support handler will provide basic canned responses.")
             self.client = None
             self.is_async = False
        else:
             # Set up OpenAI client (try async first, fallback to sync)
             if AsyncOpenAI:
                  self.client = AsyncOpenAI(api_key=self.api_key)
                  self.is_async = True
             else:
                  self.client = OpenAI(api_key=self.api_key)
                  self.is_async = False
                  logger.warning("AsyncOpenAI not found, using synchronous OpenAI client.")

        self.model = model_name

        # Common emotions in reproductive health contexts
        self.emotion_categories = {
            "anxiety": ["anxious", "nervous", "worried", "scared", "fearful", "afraid", "panic", "stressed", "terrified", "frightened", "uneasy"],
            "sadness": ["sad", "grief", "loss", "mourning", "devastated", "heartbroken", "depressed", "unhappy", "upset", "miserable", "cry", "crying", "tears"],
            "confusion": ["confused", "uncertain", "unsure", "undecided", "torn", "conflicted", "overwhelmed", "lost", "don't know what to do"],
            "shame/guilt": ["ashamed", "embarrassed", "humiliated", "guilt", "guilty", "regret", "blame", "bad person", "feel bad"],
            "hope": ["hopeful", "optimistic", "excited", "looking forward", "positive", "encouraged"],
            "anger": ["angry", "frustrated", "mad", "annoyed", "upset", "resentful", "bitter", "irritated"],
            "loneliness": ["alone", "lonely", "isolated", "no one to talk to"],
            "desperation": ["desperate", "hopeless", "helpless", "nowhere to turn"]
        }


        # Common emotional support resources for citations
        self.support_resources = [
            {
                "id": "all-options",
                "source": "All-Options Talkline",
                "url": "https://www.all-options.org/find-support/talkline/",
                "title": "All-Options Talkline - Judgment-Free Support",
                "authors": ["All-Options"] # Simplified
            },
            {
                 "id": "exhale",
                "source": "Exhale Pro-Voice",
                "url": "https://exhaleprovoice.org/",
                "title": "Exhale Pro-Voice - After-Abortion Support",
                "authors": ["Exhale Pro-Voice"] # Simplified
            },
            {
                 "id": "naf-hotline",
                "source": "National Abortion Federation Hotline", # Changed name slightly for clarity
                "url": "https://prochoice.org/patients/naf-hotline/",
                "title": "NAF Hotline - Financial & Emotional Support",
                "authors": ["National Abortion Federation"] # Simplified
            },
            {
                 "id": "apa-rh",
                "source": "APA Reproductive Health", # Changed name slightly
                "url": "https://www.apa.org/topics/reproductive-health",
                "title": "Mental Health and Reproductive Decisions",
                "authors": ["American Psychological Association"] # Simplified
            }
        ]


        # Refined support prompt template
        self.support_prompt_template = """You are Abby, a compassionate and empathetic reproductive health assistant. Your primary role right now is to provide emotional support.

User message: {query}
Detected emotions: {detected_emotions_str}
Conversation history (last few messages):
{history}

Guidelines for your response:
1.  **Acknowledge and Validate:** Directly acknowledge the user's feelings using empathetic language (e.g., "It sounds incredibly tough to feel this way...", "It makes perfect sense that you're feeling this way...", "Thank you for sharing that, it takes courage..."). Validate that their feelings are normal and okay.
2.  **Show Understanding:** Briefly reflect understanding of why they might feel that way, based on the context, without making assumptions.
3.  **Offer Support:** Express your presence and willingness to listen non-judgmentally (e.g., "I'm here to listen without judgment.", "Please know you're not alone in feeling this way.").
4.  **Keep it Focused:** Center the response on emotional validation and support. Avoid immediately jumping to solutions or factual information unless the user explicitly asks for it *in this message* or if it feels like a natural and gentle next step after validating feelings.
5.  **Gentle Resource Suggestion (Optional):** If appropriate (e.g., user expresses significant distress, loneliness, or mentions needing to talk), you can gently mention ONE relevant support resource like a talkline (e.g., "If talking to someone could help, the All-Options talkline offers judgment-free support."). Avoid overwhelming them with resources initially.
6.  **Tone:** Maintain a warm, caring, patient, and non-judgmental tone throughout. Use "I" statements to express empathy (e.g., "I hear how difficult this is...").
7.  **Conciseness:** Keep the response focused and not overly long. Aim for genuine connection over excessive text.
8.  **DO NOT:** Minimize feelings, give medical advice, make definitive statements about their situation, or push any specific agenda.

Generate a supportive and validating response based *only* on the user's message and detected emotions.
"""


    async def process_query(self, query: str, full_message: str = None, conversation_history: List[Dict[str, Any]] = None, user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        start_time = time.time()
        full_message = full_message or query
        logger.info(f"Processing emotional support query: {query[:100]}...")

        # Default empty response structure
        response_structure = {
            "text": "I'm here to listen and support you. Would you like to talk about what's on your mind?",
            "primary_content": "I'm here to listen and support you. Would you like to talk about what's on your mind?",
            "detected_emotions": [],
            "citations": [],
            "citation_objects": [],
            "processing_time": 0.0,
            "aspect_type": "emotional",
            "question_answered": True, # Assume we address the emotional need
            "needs_state_info": False
        }

        try:
            # Clean the query text
            query = query.strip()
            if not query:
                response_structure["processing_time"] = time.time() - start_time
                return response_structure

            # Detect emotions in the query
            detected_emotions = self._detect_emotions(query)
            response_structure["detected_emotions"] = detected_emotions

            # If no OpenAI client, return a canned response if emotions detected
            if not self.client:
                if detected_emotions:
                     logger.warning("OpenAI client unavailable, providing canned emotional response.")
                     canned_responses = [
                          "It sounds like you're going through a lot right now. Please know your feelings are valid and you're not alone.",
                          "I hear that this is a difficult time for you. Remember to be kind to yourself.",
                          "Thank you for sharing. It takes strength to talk about these feelings. I'm here to listen.",
                          "It's completely understandable to feel this way. Please know that support is available."
                     ]
                     response_structure["text"] = random.choice(canned_responses)
                     response_structure["primary_content"] = response_structure["text"]
                     # Add default citations even for canned response if emotions detected
                     citation_objects = self._select_relevant_citations(detected_emotions, query)
                     response_structure["citation_objects"] = citation_objects
                     response_structure["citations"] = [c["source"] for c in citation_objects]
                else:
                     # If no emotions and no client, return the default empty message
                     response_structure["text"] = "" # Make it empty so composer can handle it
                     response_structure["primary_content"] = ""

                response_structure["processing_time"] = time.time() - start_time
                return response_structure


            # Format history for prompt
            formatted_history = self._format_conversation_history(conversation_history)

            # Create the prompt
            prompt = self.support_prompt_template.format(
                query=query,
                detected_emotions_str=", ".join(detected_emotions) if detected_emotions else "None detected",
                history=formatted_history
            )

            # Generate response using OpenAI
            completion = None
            try:
                 if self.is_async:
                      completion = await self.client.chat.completions.create(
                           model=self.model,
                           messages=[{"role": "user", "content": prompt}],
                           temperature=0.7, # Moderate temperature for empathetic but grounded responses
                           max_tokens=300 # Limit token usage
                      )
                 else:
                      # Sync fallback
                      completion = self.client.chat.completions.create(
                           model=self.model,
                           messages=[{"role": "user", "content": prompt}],
                           temperature=0.7,
                           max_tokens=300
                      )
            except Exception as openai_error:
                 logger.error(f"OpenAI API call failed in EmotionalSupportHandler: {openai_error}", exc_info=True)
                 # Use canned response on API error
                 response_structure["text"] = "I understand this may be a difficult time. Please know support resources are available if you need someone to talk to."
                 response_structure["primary_content"] = response_structure["text"]
                 if detected_emotions: # Still add citations if emotions were detected before API call failed
                      citation_objects = self._select_relevant_citations(detected_emotions, query)
                      response_structure["citation_objects"] = citation_objects
                      response_structure["citations"] = [c["source"] for c in citation_objects]
                 response_structure["processing_time"] = time.time() - start_time
                 return response_structure

            # Extract and clean the response text
            raw_response_text = completion.choices[0].message.content.strip()

            # Basic cleaning (remove potential redundant sign-offs if any slip through)
            cleaned_text = re.sub(r'\n\s*(I hope this helps|Let me know if you need more|Take care).*', '', raw_response_text, flags=re.IGNORECASE).strip()

            response_structure["text"] = cleaned_text
            response_structure["primary_content"] = cleaned_text # Set primary content

            # --- FIX: Select citations ONLY if relevant emotions were detected AND response suggests resources ---
            # Only add citations if emotions were detected and the generated response hints at needing resources
            # This avoids adding citations to simple acknowledgements.
            resource_keywords = ["talkline", "resource", "support line", "hotline", "counseling", "speak to someone", "reach out"]
            suggests_resources = any(keyword in cleaned_text.lower() for keyword in resource_keywords)

            if detected_emotions and suggests_resources:
                citation_objects = self._select_relevant_citations(detected_emotions, query)
                response_structure["citation_objects"] = citation_objects
                response_structure["citations"] = [c["source"] for c in citation_objects]
                logger.info(f"Selected {len(citation_objects)} relevant citations based on detected emotions and response content.")
            else:
                # Ensure citations remain empty if no emotions or no resource mention in response
                response_structure["citation_objects"] = []
                response_structure["citations"] = []
                if detected_emotions:
                     logger.info("Emotions detected, but response did not suggest resources. Skipping citations.")
                else:
                     logger.info("No specific emotions detected, skipping emotional support citations.")
            # --- End Citation Fix ---

            response_structure["processing_time"] = time.time() - start_time
            logger.info(f"Emotional support response generated in {response_structure['processing_time']:.2f} seconds")

            return response_structure

        except Exception as e:
            logger.error(f"Unexpected error in emotional support handler: {str(e)}", exc_info=True)
            response_structure["text"] = "I'm here to listen and support you. It sounds like things might be difficult right now. Please know you're not alone. 💜" # More empathetic fallback
            response_structure["primary_content"] = response_structure["text"]
            response_structure["processing_time"] = time.time() - start_time
            # Ensure citations are empty on error
            response_structure["citations"] = []
            response_structure["citation_objects"] = []
            return response_structure

    def _detect_emotions(self, text: str) -> List[str]:
        """
        Detect emotions expressed in the text using keywords.

        Args:
            text (str): The text to analyze

        Returns:
            List[str]: List of detected emotion categories (e.g., "anxiety", "sadness")
        """
        if not text: return []
        text_lower = text.lower()
        detected = set()

        # Simple keyword-based emotion detection
        for category, keywords in self.emotion_categories.items():
            # Use regex for whole word matching to avoid partial matches
            for keyword in keywords:
                 # Handle multi-word keywords and single words appropriately
                 if ' ' in keyword: # Multi-word phrase
                     pattern = r'\b' + re.escape(keyword) + r'\b'
                 else: # Single word
                     pattern = r'\b' + re.escape(keyword) + r'\b'

                 if re.search(pattern, text_lower):
                      detected.add(category)
                      break # Move to next category once one keyword matches

        logger.debug(f"Detected emotions: {list(detected)}")
        return list(detected)

    def _format_conversation_history(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format conversation history for inclusion in the support prompt

        Args:
            conversation_history (Optional[List[Dict[str, Any]]]): Previous conversation messages

        Returns:
            str: Formatted conversation history string
        """
        if not conversation_history:
            return "No previous conversation."

        # Limit to last 3-4 messages for context
        recent_history = conversation_history[-4:]

        formatted_history = []
        for msg in recent_history:
             # Determine role, preferring 'role' key then 'sender'
             role_key = msg.get("role", msg.get("sender", "unknown"))
             role = role_key.capitalize() if role_key else "Unknown"

             # Prefer 'content' key, fallback to 'message'
             content = msg.get("content", msg.get("message", "[message content unavailable]"))
             # Limit message length to avoid overly long history context
             content_short = (content[:150] + '...') if len(content) > 150 else content
             formatted_history.append(f"{role}: {content_short}")

        return "\n".join(formatted_history) if formatted_history else "No recent messages."

    def _select_relevant_citations(self, detected_emotions: List[str], query_text: str) -> List[Dict[str, Any]]:
        """
        Select relevant citation resources based on detected emotions and query content.
        Prioritizes talklines and general support unless specific needs (like grief) are clear.

        Args:
            detected_emotions (List[str]): List of detected emotion categories
            query_text (str): The query text to analyze for relevant topics

        Returns:
            List[Dict[str, Any]]: List of citation objects relevant to the query (max 2)
        """
        if not detected_emotions:
            return [] # No citations if no emotions detected

        citations_to_add = []
        query_lower = query_text.lower()

        # Priority mapping: Emotion category -> list of preferred resource IDs
        # Order within each list indicates preference
        priority_map = {
            "sadness": ["exhale", "all-options", "naf-hotline"], # Prioritize after-abortion/loss support
            "shame/guilt": ["exhale", "all-options", "naf-hotline"], # Similar needs to sadness
            "loneliness": ["all-options", "naf-hotline", "exhale"], # Prioritize talklines
            "desperation": ["naf-hotline", "all-options"], # Prioritize hotline with potential financial aid
            "anxiety": ["all-options", "apa-rh", "naf-hotline"],
            "confusion": ["all-options", "apa-rh", "naf-hotline"],
            "anger": ["apa-rh", "all-options", "naf-hotline"], # APA might cover coping mechanisms
            "hope": [] # Typically don't need support citations for hope
        }

        # Default priority if no specific match
        default_priority = ["all-options", "naf-hotline", "apa-rh", "exhale"]

        # Find the highest priority emotion present
        resource_ids_ordered = default_priority
        # Use the order of emotions as detected if multiple, prioritize more severe ones first?
        # Simple approach: Use the first detected emotion's priority list if available.
        for emotion in detected_emotions:
            if emotion in priority_map and priority_map[emotion]: # Check if list is not empty
                 resource_ids_ordered = priority_map[emotion]
                 logger.debug(f"Using priority list for detected emotion: {emotion}")
                 break # Use the first matched priority

        # Add citations based on the ordered list, ensuring uniqueness
        added_ids = set()
        for resource_id in resource_ids_ordered:
             if len(citations_to_add) >= 2: break # Limit to 2 citations

             # Find the resource object by ID
             resource = next((res for res in self.support_resources if res.get("id") == resource_id), None)
             if resource and resource_id not in added_ids:
                  citations_to_add.append(resource)
                  added_ids.add(resource_id)


        # If still less than 2 citations, add from default list, ensuring uniqueness
        if len(citations_to_add) < 2:
             for resource_id in default_priority:
                  if len(citations_to_add) >= 2: break
                  resource = next((res for res in self.support_resources if res.get("id") == resource_id), None)
                  if resource and resource_id not in added_ids:
                       citations_to_add.append(resource)
                       added_ids.add(resource_id)


        # Ensure the final objects have required keys (id, source, url, title)
        final_citations = []
        for cit in citations_to_add:
             if all(k in cit for k in ["id", "source", "url", "title"]):
                  final_citations.append({
                       "id": cit["id"],
                       "source": cit["source"],
                       "url": cit["url"],
                       "title": cit["title"],
                       "accessed_date": datetime.now().strftime('%Y-%m-%d') # Add accessed date
                       # "authors": cit.get("authors", []) # Optionally include authors
                  })

        logger.info(f"Selected {len(final_citations)} citations: {[c['id'] for c in final_citations]}")
        return final_citations

    async def _generate_emotional_response(self, query: str, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Generate an emotional support response using OpenAI.
        Args:
            query (str): The user's query.
            conversation_history (List[Dict[str, Any]]): Conversation history.
        Returns:
            str: The generated response.
        """
        try:
            prompt = self._format_emotional_prompt(query, conversation_history)
            
            if not self.client:
                return self._get_fallback_emotional_response()
            
            try:
                if self.is_async:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,  # Slightly higher temp for emotional responses
                        max_tokens=600  # Allow longer responses for emotional support
                    )
                else:
                    # Fallback to sync client
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=600
                    )
                
                response = completion.choices[0].message.content.strip()
                
                # ***** FIX: Enhanced text cleanup *****
                # Remove common model sign-offs and filler completely
                cleaned_response = re.sub(
                    r'\n\s*(Please let me know if you need any other support|'
                    r'I hope this helps|'
                    r'If you have any more questions|'
                    r'Remember, I\'m here for you|'
                    r'Feel free to share more|'
                    r'Is there anything else|'
                    r'I\'m here to listen|'
                    r'Take care of yourself|'
                    r'You\'re not alone).*$', 
                    '', 
                    response, 
                    flags=re.DOTALL | re.IGNORECASE
                )
                
                # Remove all markdown blockquote markers ('>') at the start of lines or after HTML tags
                cleaned_response = re.sub(r'^>\s*', '', cleaned_response, flags=re.MULTILINE)
                cleaned_response = re.sub(r'\n>\s*', '\n', cleaned_response)
                cleaned_response = re.sub(r'(<[^>]+>)\s*>\s*', r'\1', cleaned_response)
                
                # Remove any remaining blockquote markers that might be in the middle of text
                cleaned_response = cleaned_response.replace(' > ', ' ')
                
                # Clean up double spaces and extra whitespace
                cleaned_response = re.sub(r'\s{2,}', ' ', cleaned_response)
                cleaned_response = cleaned_response.strip()
                # ***** END FIX *****
                
                return cleaned_response
                
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return self._get_fallback_emotional_response()
                
        except Exception as e:
            logger.error(f"Error generating emotional response: {str(e)}")
            return self._get_fallback_emotional_response()


# --- END OF FILE emotional_support_handler.py ---