import logging
import uuid
import time
from typing import Dict, List, Any, Optional
import datetime
import re
import os
import asyncio
# Use AsyncOpenAI if available
try:
    from openai import AsyncOpenAI
    USE_ASYNC_OPENAI = True
except ImportError:
    from openai import OpenAI
    USE_ASYNC_OPENAI = False

logger = logging.getLogger(__name__)

class ResponseComposer:
    """
    Composes final responses by intelligently blending outputs from different specialized handlers
    """
    
    def __init__(self):
        """Initialize the response composer"""
        logger.info("Initializing ResponseComposer")
        self.openai_client = None
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                # Try async first
                if USE_ASYNC_OPENAI:
                    self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                    logger.info("AsyncOpenAI client initialized for response composer")
                else:
                    # Fallback to sync
                    self.openai_client = OpenAI(api_key=openai_api_key)
                    logger.info("OpenAI (sync) client initialized for response composer")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
        else:
            logger.warning("OpenAI API key not found. Response composer will use basic formatting.")
        
        # Define transition phrases for different aspect combinations
        self.transitions = {
            # Knowledge to Emotional
            "knowledge_emotional": [
                "In addition to this information, I want to acknowledge that ",
                "Beyond the facts, it's also important to consider that ",
                "While those are the medical facts, I understand that "
            ],
            
            # Knowledge to Policy
            "knowledge_policy": [
                "Regarding the legal aspects of this topic, ",
                "As for the policy implications, ",
                "In terms of the regulations in your area, "
            ],
            
            # Emotional to Knowledge
            "emotional_knowledge": [
                "To give you some additional information on this topic, ",
                "Here are some facts that might be helpful: ",
                "From a medical perspective, "
            ],
            
            # Emotional to Policy
            "emotional_policy": [
                "Regarding the legal situation, ",
                "As for the policies in your area, ",
                "In terms of the regulations that might affect you, "
            ],
            
            # Policy to Knowledge
            "policy_knowledge": [
                "Beyond the legal aspects, here's some general information: ",
                "In addition to the policy information, you might want to know that ",
                "From a medical perspective, "
            ],
            
            # Policy to Emotional
            "policy_emotional": [
                "I understand this information can bring up feelings. ",
                "Many people experience various emotions when considering these policies. ",
                "While those are the regulations, I recognize that this can be a complex emotional topic. "
            ]
        }
        
        # Define default transitions
        self.default_transition = "Additionally, "

    async def compose_response(self, message: str, aspect_responses: List[Dict[str, Any]],
                        classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compose a coherent HTML response from multiple aspect handlers.
        """
        start_time = time.time()
        logger.info(f"Composing response from {len(aspect_responses)} aspect(s)")

        if not aspect_responses:
            logger.warning("No aspect responses provided to composer.")
            return self._create_fallback_response("No content generated.") # Pass specific message

        final_html_content = ""
        all_citations = []
        all_citation_objects = []
        processed_aspect_types = set()
        special_attributes = {}

        # --- FIX: Prioritize and structure aspect processing ---
        # Preferred order: Policy -> Knowledge -> Emotional
        aspect_order = ['policy', 'knowledge', 'emotional']
        ordered_responses = sorted(aspect_responses, key=lambda r: aspect_order.index(r.get('aspect_type', 'knowledge')) if r.get('aspect_type') in aspect_order else 99)

        for i, resp in enumerate(ordered_responses):
            aspect_type = resp.get('aspect_type', 'knowledge')
            primary_content = resp.get('primary_content', resp.get('text', '')).strip()

            if not primary_content:
                logger.debug(f"Skipping empty response for aspect type: {aspect_type}")
                continue

            # --- FIX: Prevent processing the same aspect type twice (e.g., if decomposition failed) ---
            if aspect_type in processed_aspect_types:
                 logger.warning(f"Skipping duplicate aspect type: {aspect_type}")
                 continue
            processed_aspect_types.add(aspect_type)

            # --- FIX: Handle Policy content definitively ---
            # Use policy handler's primary content directly if it exists and is good
            # The _format_policy_details can serve as a fallback within the handler if needed
            if aspect_type == 'policy':
                 # The PolicyHandler's process_query should return the final desired text
                 # in 'primary_content'. We trust that content here.
                 logger.debug("Using policy aspect's primary_content directly.")
                 # Special attributes are collected later

            # Add transition phrase if needed (skip for the first aspect)
            # Use simple paragraph breaks now, OpenAI refinement can improve flow
            if i > 0:
                final_html_content += "\n\n" # Simple separator, let OpenAI handle transitions

            # Append the primary content of the current aspect
            final_html_content += primary_content

            # --- FIX: Collect Citations Correctly ---
            # Ensure citation_objects is a list of dicts
            citations_from_resp = resp.get('citation_objects', [])
            if isinstance(citations_from_resp, list):
                 for cit_obj in citations_from_resp:
                      if isinstance(cit_obj, dict) and 'url' in cit_obj:
                           # Use URL as key for deduplication
                           if cit_obj['url'] not in [c.get('url') for c in all_citation_objects if c.get('url')]:
                                all_citation_objects.append(cit_obj)
                      elif isinstance(cit_obj, dict) and 'source' in cit_obj: # Handle objects without URL
                           if cit_obj['source'] not in [c.get('source') for c in all_citation_objects if not c.get('url')]:
                                all_citation_objects.append(cit_obj)
            # Collect string citations only if object is missing (should be rare now)
            string_citations = resp.get('citations', [])
            if isinstance(string_citations, list):
                 for cit_str in string_citations:
                      if isinstance(cit_str, str) and cit_str not in [c.get('source') for c in all_citation_objects]:
                            # If we don't have an object for this source string yet, add a basic one
                            all_citation_objects.append({
                                "source": cit_str,
                                "title": cit_str,
                                "url": None,
                                "accessed_date": datetime.datetime.now().strftime('%Y-%m-%d')
                            })

            # Collect special attributes from PolicyHandler
            if aspect_type == "policy":
                for key in ["state_code", "state_name", "is_legal", "travel_state_code", "travel_state_name", "requires_travel", "nearby_states", "map_data", "show_map", "zip_code", "policy_details"]:
                     if key in resp:
                          special_attributes[key] = resp[key]

        # --- FIX: Apply final formatting and cleaning ---
        # Now properly await the async method
        final_html = await self._apply_final_formatting_and_cleanup(final_html_content)

        # Extract unique source strings for the 'citations' field (legacy compatibility?)
        all_citations = list(set(c.get('source', 'Unknown') for c in all_citation_objects))

        # Create the final response object
        response = {
            "text": final_html, # Final HTML content
            "citations": all_citations, # List of source names
            "citation_objects": all_citation_objects, # List of citation dicts
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "session_id": classification.get("session_id", str(uuid.uuid4())), # Pass session_id if available
            "graphics": special_attributes.pop("graphics", []), # Handle graphics separately
            **special_attributes # Add policy-specific fields like state_code, map_data, etc.
        }

        # Ensure all required fields exist, using defaults if necessary
        response.setdefault("processing_time", time.time() - start_time)
        response.setdefault("show_map", False) # Default map state
        response.setdefault("map_data", None)
        response.setdefault("zip_code", None)
        response.setdefault("state_code", None)
        response.setdefault("state_name", None)
        response.setdefault("is_legal", None)
        response.setdefault("travel_state_code", None)
        response.setdefault("travel_state_name", None)
        response.setdefault("requires_travel", None)
        response.setdefault("nearby_states", None)
        response.setdefault("policy_details", None)

        logger.info(f"Response composition finished in {response['processing_time']:.2f}s")
        return response

    async def _apply_final_formatting_and_cleanup(self, combined_text: str) -> str:
        """Applies final HTML formatting, cleans artifacts, and uses OpenAI for refinement."""
        if not combined_text:
            return ""

        logger.debug("Starting final formatting and cleanup")

        # --- FIX: Step 1: Pre-cleaning - Remove blockquote markers ---
        # First, remove stray > characters at beginning of lines and mid-line
        cleaned_text = re.sub(r'^(>|>)\s*', '', combined_text, flags=re.MULTILINE).strip()
        cleaned_text = cleaned_text.replace(' > ', ' ') # Catch mid-line markers
        logger.debug("Removed blockquote markers")

        # --- Step 2: Convert specific markers to basic HTML structure ---
        # Convert ###TITLE### markers from KnowledgeHandler to <h3>
        cleaned_text = re.sub(r'^\s*###(.*?)###\s*\n?', r'<h3>\1</h3>\n', cleaned_text, flags=re.MULTILINE)
        logger.debug("Converted ###TITLE### to <h3>")

        # Convert Markdown bold **text** to <strong>
        cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cleaned_text)
        logger.debug("Converted ** to <strong>")

        # Convert Markdown • bullets to simple <li> (will be wrapped later)
        # Ensure each bullet starts on a new line conceptually before converting
        cleaned_text = re.sub(r'\n\s*•\s+', '\n<li>', cleaned_text)
        cleaned_text = re.sub(r'^•\s+', '<li>', cleaned_text) # Handle first line
        # Also handle * bullets
        cleaned_text = re.sub(r'\n\s*\*\s+', '\n<li>', cleaned_text)
        cleaned_text = re.sub(r'^\*\s+', '<li>', cleaned_text) # Handle first line
        # Add closing tag, assuming list items end at newline or next list item
        cleaned_text = re.sub(r'(<li>.*?)(?=\n<li>|\n\n|\Z)', r'\1</li>', cleaned_text, flags=re.DOTALL)
        logger.debug("Converted bullets to <li>")

        # --- Step 3: Basic HTML Structure (Paragraphs and Lists) ---
        final_html_parts = []
        current_list_items = []
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('<li>') and line.endswith('</li>'):
                 # Remove <li></li> tags for direct insertion
                 current_list_items.append(line[4:-5].strip())
            else:
                 # If we were building a list, close it first
                 if current_list_items:
                      final_html_parts.append('<ul class="chat-bullet-list">' + ''.join(f'<li>{item}</li>' for item in current_list_items) + '</ul>')
                      current_list_items = []
                 # Add non-list line (could be <h3> or plain text)
                 if line.startswith('<h3>') and line.endswith('</h3>'):
                      final_html_parts.append(line)
                 else:
                      # Wrap plain text in paragraphs
                      final_html_parts.append(f'<p class="message-paragraph">{line}</p>')

        # Close any pending list
        if current_list_items:
             final_html_parts.append('<ul class="chat-bullet-list">' + ''.join(f'<li>{item}</li>' for item in current_list_items) + '</ul>')

        structured_html = "\n".join(final_html_parts)
        logger.debug("Applied basic HTML structure (p, h3, ul, li)")
        
        # Save a copy of the original structured HTML before refinement
        original_structured_html = structured_html
        
        # --- FIX: Check for key policy information ---
        # Flag indicating if this appears to be a policy response (look for common policy markers)
        is_policy_response = False
        if re.search(r'(abortion|legal|waiting period|gestational|minor|parental)', original_structured_html, re.IGNORECASE):
            is_policy_response = True
            logger.debug("Detected policy-related content in the response")

        # Use OpenAI to refine the text while preserving the HTML structure
        if self.openai_client:
            try:
                logger.info("Using OpenAI to refine final HTML content")
                refined_html = await self._refine_with_openai(structured_html)
                
                # --- FIX: Ensure policy content is preserved ---
                if is_policy_response:
                    # Check if the refined text still has the key policy information by doing a basic pattern match
                    key_policy_patterns = [
                        r'(abortion|termination).*?(legal|available|restricted|banned)',
                        r'gestational limit',
                        r'waiting period',
                        r'minor|parental'
                    ]
                    
                    original_has_patterns = sum(1 for pattern in key_policy_patterns 
                                               if re.search(pattern, original_structured_html, re.IGNORECASE))
                    refined_has_patterns = sum(1 for pattern in key_policy_patterns 
                                              if re.search(pattern, refined_html, re.IGNORECASE))
                    
                    # If the refinement lost significant policy information, fall back to original
                    if original_has_patterns > refined_has_patterns:
                        logger.warning(f"OpenAI refinement lost policy information (original: {original_has_patterns} patterns, refined: {refined_has_patterns} patterns). Using original content.")
                        return original_structured_html
                
                return refined_html
            except Exception as e:
                logger.error(f"Error refining text with OpenAI: {str(e)}")
                # Return the basic structured content on error
                return structured_html
        else:
            # If no OpenAI client available, return the basic structured content
            return structured_html

    def _create_fallback_response(self, error_context: str = "general error") -> Dict[str, Any]:
         """Create a fallback response when something goes wrong"""
         logger.error(f"Creating fallback response due to: {error_context}")
         message_id = str(uuid.uuid4())
         # Use only AbortionFinder resource for generic fallback/error
         fallback_citations = [
             {
                 "id": "abortionfinder-fallback",
                 "source": "AbortionFinder",
                 "title": "Find Verified Abortion Care",
                 "url": "https://www.abortionfinder.org/",
                 "accessed_date": datetime.datetime.now().strftime('%Y-%m-%d')
             }
         ]
         return {
             "text": "<p class='message-paragraph'>I apologize, but I encountered an issue and couldn't generate a complete response. You might find helpful information at AbortionFinder.org.</p>",
             "citations": [c['source'] for c in fallback_citations],
             "citation_objects": fallback_citations,
             "message_id": message_id,
             "timestamp": time.time(),
             "session_id": str(uuid.uuid4()), # Generate a new one for error? Or use context?
             "graphics": [],
             "processing_time": 0.1, # Indicate minimal processing time
             "show_map": False,
             "metadata": {"error": True, "error_context": error_context}
         }

    def _get_transition(self, transition_key: str) -> str:
        """
        Get a transition phrase for connecting different aspects
        
        Args:
            transition_key (str): Key in the format "from_to" (e.g., "knowledge_policy")
            
        Returns:
            str: A transition phrase
        """
        if transition_key in self.transitions:
            transitions = self.transitions[transition_key]
            return transitions[int(time.time()) % len(transitions)]
        return self.default_transition 

    async def _refine_with_openai(self, structured_html: str) -> str:
        """Use OpenAI to refine the HTML content while preserving structure.
        
        Args:
            structured_html (str): The HTML content to refine
            
        Returns:
            str: The refined HTML content
        """
        # Refined system prompt emphasizing HTML preservation and policy details
        system_prompt = """You are an expert text editor specializing in refining chatbot responses for reproductive health.
Your task is to improve the flow, empathy, and clarity of the provided HTML content **while preserving ALL HTML tags** (e.g., <p>, <h3>, <ul>, <li>, <strong>).

CRITICAL POLICY DETAILS: When refining abortion law information, you MUST preserve:
1. Exact gestational limits (weeks or LMP values)
2. Legal status (banned/legal/restricted)
3. Waiting period lengths (hours/days)
4. Parental notification/consent requirements for minors
5. Insurance coverage details

General Refinement Guidelines:
- Ensure a warm, supportive, and non-judgmental tone.
- Improve sentence flow and transitions between paragraphs or sections.
- Maintain factual accuracy. Do not add or remove information, only rephrase for clarity and empathy.
- Remove any repetitive phrasing or redundant sentences.
- Ensure the language is accessible and avoids overly clinical terms where appropriate.

Format Requirements:
- **CRITICAL: Return ONLY the improved HTML content. Do not add any introductory text, explanations, or comments outside the HTML structure.**
- **Make sure all original HTML tags are present in the output.**
- Preserve all numeric values, especially those related to weeks, waiting periods, or other legal requirements.
"""
        max_retries = 2
        completion = None # Initialize completion to None

        for attempt in range(max_retries):
            logger.debug(f"OpenAI refinement attempt {attempt + 1}")
            try:
                # ***** FIX START: Correctly handle async/sync client *****
                if USE_ASYNC_OPENAI and hasattr(self.openai_client, 'chat'):
                    # Await the async call directly instead of using asyncio.run()
                    logger.debug("Using AsyncOpenAI client...")
                    completion = await self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": structured_html}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                elif not USE_ASYNC_OPENAI and hasattr(self.openai_client, 'chat'):
                    # Run the synchronous call (no await needed)
                    logger.debug("Using synchronous OpenAI client...")
                    completion = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": structured_html}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                else:
                    logger.error("OpenAI client structure not recognized or client is None.")
                    completion = None # Ensure completion is None if client is unusable
                # ***** FIX END *****
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                completion = None

            if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
                improved_html = completion.choices[0].message.content.strip()
                
                # Basic validation: Check if it looks like HTML
                if improved_html.startswith('<') and improved_html.endswith('>'):
                    logger.info("OpenAI refinement successful.")
                    
                    # --- NEW: Policy detail preservation check ---
                    # Extract and verify critical policy details haven't been altered
                    # Check for gestational limits (weeks)
                    weeks_pattern = r'(\d+)\s*(?:weeks|week)'
                    original_weeks = re.findall(weeks_pattern, structured_html)
                    refined_weeks = re.findall(weeks_pattern, improved_html)
                    
                    # Check for waiting periods (hours)
                    hours_pattern = r'(\d+)\s*(?:hours|hour)'
                    original_hours = re.findall(hours_pattern, structured_html)
                    refined_hours = re.findall(hours_pattern, improved_html)
                    
                    # Check for banned/legal status
                    legal_terms = ['banned', 'illegal', 'prohibited', 'restricted', 'legal']
                    has_policy_details = any(term in structured_html.lower() for term in legal_terms)
                    
                    # Verify critical elements preserved
                    critical_detail_lost = False
                    
                    if has_policy_details:
                        if sorted(original_weeks) != sorted(refined_weeks):
                            logger.warning(f"Gestational weeks changed: {original_weeks} → {refined_weeks}")
                            critical_detail_lost = True
                            
                        if sorted(original_hours) != sorted(refined_hours):
                            logger.warning(f"Waiting period hours changed: {original_hours} → {refined_hours}")
                            critical_detail_lost = True
                            
                        # Check if key policy terms are missing
                        for term in legal_terms:
                            if term in structured_html.lower() and term not in improved_html.lower():
                                logger.warning(f"Legal status term '{term}' lost in refinement")
                                critical_detail_lost = True
                    
                    if critical_detail_lost:
                        logger.warning("Critical policy details were altered. Using original content for safety.")
                        return structured_html
                    # --- END NEW: Policy detail preservation check ---
                    
                    # --- FIX: Final cleanup after OpenAI ---
                    # Remove any stray '>' again, just in case
                    improved_html = re.sub(r'^(>|>)\s*', '', improved_html, flags=re.MULTILINE)
                    # Remove potential empty tags introduced
                    improved_html = re.sub(r'<p[^>]*>\s*<\/p>', '', improved_html)
                    improved_html = re.sub(r'<li[^>]*>\s*<\/li>', '', improved_html)
                    return improved_html.strip()
                else:
                    logger.warning(f"OpenAI refinement produced non-HTML output (attempt {attempt + 1}): {improved_html[:100]}...")
            else:
                logger.warning(f"OpenAI refinement failed or returned empty response (attempt {attempt + 1}).")

            # If retry is needed, wait briefly using asyncio.sleep for non-blocking delay
            if attempt < max_retries - 1:
                await asyncio.sleep(1) # Use asyncio.sleep instead of time.sleep

        # If all retries fail, fall back to the structured HTML
        logger.warning("OpenAI refinement failed after retries, returning structured HTML.")
        return structured_html 