# --- START OF FILE policy_api.py ---

import os
import logging
import requests
import json
import re
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class PolicyAPI:
    """
    Integration with the abortion policy API to provide up-to-date policy information.
    Focuses on fetching data; formatting is primarily handled by PolicyHandler or ResponseComposer.
    """

    def __init__(self):
        """Initialize the Policy API"""
        logger.info("Initializing PolicyAPI Client")
        # Default API key (will be overridden by environment variable if available)
        default_api_key = ''

        # Try to get API key from environment vars
        self.api_key = os.environ.get('ABORTION_POLICY_API_KEY', default_api_key)
        self.base_url = os.environ.get("POLICY_API_BASE_URL", "https://api.abortionpolicyapi.com/v1") # Allow override

        if self.api_key:
            logger.info("Abortion Policy API key found")
        else:
            logger.warning("No Abortion Policy API key found in environment - API calls will likely fail")

        # Define US state codes and names
        self.STATE_NAMES = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }

        # Create lowercase state names for searches
        self.STATE_NAMES_LOWER = {k.lower(): k for k in self.STATE_NAMES.keys()}
        for k, v in self.STATE_NAMES.items():
            if v: # Ensure state name is not empty
                self.STATE_NAMES_LOWER[v.lower()] = k

        # Define endpoints to query
        self.endpoints = {
            "waiting_periods": "waiting_periods",
            "insurance_coverage": "insurance_coverage",
            "gestational_limits": "gestational_limits",
            "minors": "minors"
            # Consider adding more endpoints if the API supports them, e.g.,
            # "exceptions": "exceptions_to_abortion_bans"
        }

        # Cache for API responses to reduce calls
        self.api_cache = {}
        self.cache_ttl = 3600 * 6 # Cache for 6 hours

        logger.info("Policy API Client initialized successfully")

    def _extract_state_from_question(self, question: str) -> Optional[str]:
        """
        Extract state information from a user's question.
        Prioritizes full state names and handles ambiguity.
        (Consistent with logic in PolicyHandler)

        Args:
            question (str): User query text

        Returns:
            Optional[str]: Two-letter state code or None if not found
        """
        if not question or not isinstance(question, str): return None
        query_lower = question.lower().strip()
        if not query_lower: return None


        # --- FIX (Bug 2): Prioritize full state names ---
        # Sort by length descending to match multi-word states first
        sorted_state_names = sorted(self.STATE_NAMES.items(), key=lambda item: len(item[1]) if item[1] else 0, reverse=True)
        for state_code, state_name in sorted_state_names:
             if not state_name: continue # Skip if state name is empty
             # Use word boundaries for exact match
             state_pattern = r'\b' + re.escape(state_name.lower()) + r'\b'
             if re.search(state_pattern, query_lower):
                 logger.info(f"Found state name '{state_name}' ({state_code}) in question")
                 return state_code

        # --- FIX: Handle common state variants (Check BEFORE abbreviations) ---
        # Add more variants
        state_variants = { "cali": "CA", "calif": "CA", "ny": "NY", "fla": "FL", "penn": "PA", "indy": "IN", "tex": "TX", "wash": "WA", "mass": "MA", "colo": "CO", "conn": "CT", "mich": "MI", "minn": "MN", "miss": "MS", "wisc": "WI", "virg": "VA", "n carolina": "NC", "s carolina": "SC", "n dakota": "ND", "s dakota": "SD", "w virginia": "WV" }
        # Sort variants by length descending to match longer ones first
        sorted_variants = sorted(state_variants.items(), key=lambda item: len(item[0]), reverse=True)
        temp_query_lower = query_lower # Work on copy for variant removal
        for variant, code in sorted_variants:
             variant_pattern = r'\b' + re.escape(variant) + r'\b'
             if re.search(variant_pattern, temp_query_lower):
                  logger.info(f"Found state variant '{variant}' in question, matching to {code}")
                  # Remove matched variant to prevent shorter matches later
                  temp_query_lower = re.sub(variant_pattern, '', temp_query_lower, count=1)
                  # Return the first variant found
                  return code


        # --- FIX: Check for state abbreviations carefully ---
        words = query_lower.split()
        potential_codes = []
        # Use PolicyHandler's list for consistency
        ambiguous_abbrs_lower = {"in", "on", "at", "me", "hi", "ok", "or", "la", "pa", "no", "so", "de", "oh", "co", "wa", "va", "ma", "id", "mo", "ri", "ct", "md", "nh", "nj", "sc", "tn", "ut", "vt", "wi", "al"}


        # Check uppercase first
        for word in question.split(): # Use original case question here
             word_cleaned = re.sub(r'[^\w]', '', word) # Clean punctuation
             if len(word_cleaned) == 2 and word_cleaned.isupper() and word_cleaned in self.STATE_NAMES:
                  potential_codes.append(word_cleaned)
                  logger.info(f"Found uppercase state code '{word_cleaned}' in question")

        # Check lowercase, being careful with ambiguous ones
        if not potential_codes: # Only if no uppercase codes found
            for word_idx, word in enumerate(words):
                word_cleaned = re.sub(r'[^\w]', '', word)
                if len(word_cleaned) == 2 and word_cleaned.islower():
                    abbr_upper = word_cleaned.upper()
                    if abbr_upper in self.STATE_NAMES:
                        # Check if ambiguous
                        if word_cleaned in ambiguous_abbrs_lower:
                            # Require context for ambiguous codes
                            context_before = words[word_idx-1].lower() if word_idx > 0 else ""
                            context_after = words[word_idx+1].lower() if word_idx < len(words)-1 else ""
                            # Added more context terms
                            context_terms = {"state", "laws", "policy", "abortion", "access", "in", "for", "near", "clinic", "regulation", "ban", "legal", "travel", "visit"}
                            # Check if context words are immediately before or after
                            if context_before in context_terms or context_after in context_terms:
                                potential_codes.append(abbr_upper)
                                logger.info(f"Found ambiguous state code '{abbr_upper}' with context in question")
                            # Check if it's followed by a zip code
                            elif context_after and re.match(r'^\d{5}$', context_after):
                                 potential_codes.append(abbr_upper)
                                 logger.info(f"Found ambiguous state code '{abbr_upper}' followed by ZIP in question")
                            else:
                                logger.info(f"Skipping ambiguous lowercase code '{word_cleaned}' due to lack of immediate context")
                        else:
                            # Non-ambiguous lowercase codes are accepted
                            potential_codes.append(abbr_upper)
                            logger.info(f"Found non-ambiguous lowercase state code '{abbr_upper}' in question")

        # If exactly one potential code found, return it
        if len(potential_codes) == 1:
            return potential_codes[0]
        elif len(potential_codes) > 1:
            logger.warning(f"Multiple potential state codes found: {potential_codes}. Cannot reliably determine state from question.")
            return None # Disambiguation is hard here


        # --- FIX (Bug 2): Check for non-US countries ONLY if no US state found ---
        non_us_countries = ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany',
                           'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']
        if 'united kingdom' in query_lower: non_us_countries.append('united kingdom')

        for country in non_us_countries:
            country_pattern = r'\b' + re.escape(country) + r'\b'
            # Additional check to avoid "Mexico" matching "New Mexico"
            if country == 'mexico' and 'new mexico' in query_lower:
                 continue
            if re.search(country_pattern, query_lower):
                logger.info(f"Non-US country detected: {country}")
                return None # Return None for non-US countries

        # If no state found
        logger.debug(f"No state code found in question: '{query_lower[:100]}...'")
        return None

    def get_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Get abortion policy data for a state, using cache if possible.

        Args:
            state_code (str): The 2-letter state code.

        Returns:
            Dict[str, Any]: Dictionary containing policy data for the state or fallback data.
        """
        # Ensure state_code is uppercase and valid
        state_code = state_code.upper()
        if state_code not in self.STATE_NAMES:
             logger.error(f"Invalid state code provided to get_policy_data: {state_code}")
             return self._get_fallback_data(state_code) # Return fallback for invalid codes

        state_name = self.STATE_NAMES.get(state_code, state_code)

        # Check cache
        cache_entry = self.api_cache.get(state_code)
        if cache_entry and time.time() - cache_entry.get('timestamp', 0) < self.cache_ttl:
            logger.info(f"Returning cached policy data for {state_name} ({state_code})")
            return cache_entry['data']

        # If not in cache or expired, fetch from API
        logger.info(f"Fetching policy data for {state_name} ({state_code}) from API")
        try:
            # Ensure API key is available
            if not self.api_key:
                logger.error("Abortion Policy API key is missing. Cannot fetch data.")
                return self._get_fallback_data(state_code)

            headers = {'token': self.api_key}
            policy_info = {"endpoints": {}}
            api_base = self.base_url.rstrip('/')

            all_tasks_successful = True
            results = {}
            for key, endpoint in self.endpoints.items():
                 url = f"{api_base}/{endpoint}/states/{state_code}"
                 try:
                      # Using requests library for synchronous operation within this class
                      response = requests.get(url, headers=headers, timeout=10) # Added timeout
                      response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                      # Check content type before parsing JSON
                      content_type = response.headers.get('Content-Type', '')
                      if 'application/json' in content_type:
                           data = response.json()
                            # Handle potential empty responses or list responses from API
                           if isinstance(data, list) and len(data) > 0:
                                results[key] = data[0] if isinstance(data[0], dict) else {}
                           elif isinstance(data, dict):
                                results[key] = data if data else {}
                           else:
                                logger.warning(f"API returned unexpected JSON structure for {url}: {type(data)}")
                                results[key] = {}
                      else:
                           logger.warning(f"API returned non-JSON response for {url}. Status: {response.status}. Content-Type: {content_type}")
                           results[key] = {}
                           all_tasks_successful = False # Treat non-JSON as failure

                      logger.debug(f"Successfully fetched data for {key} from {url}")

                 except requests.exceptions.Timeout:
                      logger.error(f"Timeout fetching {key} for {state_code} from {url}")
                      results[key] = {}
                      all_tasks_successful = False
                 except requests.exceptions.HTTPError as http_err:
                      status_code = http_err.response.status_code
                      logger.error(f"HTTP error {status_code} fetching {key} for {state_code} from {url}")
                      results[key] = {}
                      # Allow 404s (no data for state/endpoint) but flag others as failures
                      if status_code != 404:
                           all_tasks_successful = False
                 except requests.exceptions.RequestException as req_err:
                      logger.error(f"Request error fetching {key} for {state_code}: {req_err}")
                      results[key] = {}
                      all_tasks_successful = False
                 except json.JSONDecodeError as json_err:
                      logger.error(f"JSON decode error fetching {key} for {state_code} from {url}: {json_err}")
                      results[key] = {}
                      all_tasks_successful = False

                 time.sleep(0.2) # Small delay between requests


            # If any critical fetch failed (non-404 error), return fallback
            if not all_tasks_successful:
                 logger.warning(f"One or more critical API calls failed for {state_code}, returning fallback data.")
                 return self._get_fallback_data(state_code)

            # Process successful results
            policy_info["endpoints"] = results
            if any(policy_info["endpoints"].values()): # Check if we got any data at all
                policy_info["state_code"] = state_code
                policy_info["state_name"] = state_name
                policy_info["error"] = False # Mark as not an error
                # Nest data under state code for consistency (important for PolicyHandler)
                formatted_endpoints = {}
                for key, data in policy_info["endpoints"].items():
                      formatted_endpoints[key] = {state_code: data} if data else {}
                policy_info["endpoints"] = formatted_endpoints

                # Cache the result
                cache_data = policy_info.copy() # Store a copy
                self.api_cache[state_code] = {'timestamp': time.time(), 'data': cache_data}
                logger.info(f"Successfully fetched and cached policy data for {state_name}")
                return policy_info
            else:
                logger.error(f"No policy data found for {state_code} despite successful API status codes.")
                # Cache the fact that no data was found (to avoid refetching quickly)
                fallback_data = self._get_fallback_data(state_code)
                fallback_data['error_message'] = "No specific data returned from API."
                self.api_cache[state_code] = {'timestamp': time.time(), 'data': fallback_data}
                return fallback_data

        except Exception as e:
            logger.error(f"Unexpected error in get_policy_data for {state_code}: {str(e)}", exc_info=True)
            return self._get_fallback_data(state_code)


    def _format_api_response(self, api_data, state_name):
        """
        Minimal formatting of raw API data. Detailed formatting is handled elsewhere.
        """
        logger.debug(f"Passing raw API data for {state_name} to handler.")
        # Return the structured data; PolicyHandler or ResponseComposer will format it.
        return api_data


    def _get_fallback_data(self, state_code_or_context: str) -> Dict[str, Any]:
        """Generate fallback policy data structure when API fails or data is invalid."""
        state_code = None
        state_name = "the requested area"
        # Try to determine state code from context if possible
        if isinstance(state_code_or_context, str) and len(state_code_or_context) == 2 and state_code_or_context.upper() in self.STATE_NAMES:
            state_code = state_code_or_context.upper()
            state_name = self.STATE_NAMES.get(state_code, state_code)
        elif isinstance(state_code_or_context, str):
             # Try extracting from the context string
             potential_code = self._extract_state_from_question(state_code_or_context)
             if potential_code:
                  state_code = potential_code
                  state_name = self.STATE_NAMES.get(state_code, state_code)
             else:
                  state_name = state_code_or_context # Use the context as the name if unresolvable

        logger.warning(f"Using fallback data structure for {state_name} (Code: {state_code})")
        # Default data structure indicating error
        return {
            'error': True,
            'state_code': state_code, # May be None if context wasn't a valid state
            'state_name': state_name,
            'error_message': f"Could not retrieve detailed policy information for {state_name}.",
            'endpoints': {}, # Ensure endpoints key exists but is empty
            'legal_status': f"Unable to retrieve specific policy information for {state_name}.", # Generic fallback text
            'gestational_limit': "Information currently unavailable.",
            'restrictions': ["Please check reliable sources like Guttmacher Institute or Planned Parenthood for current laws."],
            'services': ["Contact Planned Parenthood or visit AbortionFinder.org to find services near you."],
            'resources': [ # Keep general resources
                "Planned Parenthood",
                "National Abortion Federation",
                "AbortionFinder.org" # Updated resource
            ],
            'sources': [ # Default fallback citations
                {
                    'id': 'guttmacher-fallback',
                    'source': 'Guttmacher Institute',
                    'url': 'https://www.guttmacher.org/state-policy/explore/abortion-policy',
                    'title': 'Guttmacher State Policy Explorer',
                    'accessed_date': datetime.now().strftime('%Y-%m-%d')
                },
                 {
                    'id': 'finder-fallback',
                    'source': 'AbortionFinder.org',
                    'url': 'https://www.abortionfinder.org/',
                    'title': 'AbortionFinder - Find Care',
                    'accessed_date': datetime.now().strftime('%Y-%m-%d')
                 }
            ]
        }


    def get_state_code(self, location_context: str) -> Optional[str]:
        """
        Converts a location string (state name or abbreviation) into a 2-letter state code.
        Prioritizes full state names. Uses internal logic consistent with _extract_state_from_question.
        """
        if not location_context or not isinstance(location_context, str):
             return None

        location_lower = location_context.strip().lower()

        # Use the extraction logic directly
        state_code = self._extract_state_from_question(location_lower)

        if state_code:
            logger.info(f"Resolved location context '{location_context}' to state code: {state_code}")
        else:
            logger.warning(f"Could not resolve location context '{location_context}' to a state code.")

        return state_code


    def get_abortion_policy(self, location_context: str) -> Dict[str, Any]:
        """
        Get abortion policy information for a given location. Returns structured data.

        Args:
            location_context (str): The location to get policy for (state name or abbreviation)

        Returns:
            Dict[str, Any]: Policy data dictionary or fallback data dictionary.
        """
        try:
            if not location_context or not isinstance(location_context, str):
                 logger.error("Invalid location context provided to get_abortion_policy.")
                 return self._get_fallback_data("an unspecified location")

            location_lower = location_context.lower().strip()

            # --- FIX (Bug 2): Check state code first ---
            state_code = self.get_state_code(location_context)

            if state_code:
                logger.info(f"Getting policy data for state: {state_code}")
                return self.get_policy_data(state_code) # This now handles fetching and fallback

            # --- FIX (Bug 2): Check non-US countries AFTER US states ---
            non_us_countries = ['india', 'canada', 'uk', 'united kingdom', 'australia', 'mexico', 'france', 'germany',
                           'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']
            country_found = None
            for country in non_us_countries:
                country_pattern = r'\b' + re.escape(country) + r'\b'
                # Avoid matching "mexico" within "new mexico" which should have been caught by get_state_code
                if country == 'mexico' and 'new mexico' in location_lower:
                    continue
                if re.search(country_pattern, location_lower):
                     country_found = country.title()
                     # Handle UK variant
                     if country_found == 'Uk': country_found = 'United Kingdom'
                     break

            if country_found:
                logger.info(f"Non-US country detected: {country_found}")
                return {
                    'error': True,
                    'is_international': True,
                    'location_name': country_found,
                    'message': f"My information focuses on U.S. states. For abortion access details in {country_found}, please consult local health authorities or reliable international organizations.",
                    'sources': [ # Provide general international resources
                         {
                             'id': 'who-rh',
                             'source': 'World Health Organization',
                             'url': 'https://www.who.int/health-topics/sexual-and-reproductive-health-and-rights',
                             'title': 'WHO Sexual and Reproductive Health',
                             'accessed_date': datetime.now().strftime('%Y-%m-%d')
                         }
                    ]
                }

            # If neither state nor known non-US country, return specific fallback
            logger.info(f"Could not recognize location: {location_context}")
            return {
                'error': True,
                'location_unrecognized': True,
                'message': f"I'm sorry, I couldn't recognize '{location_context}' as a U.S. state or known country. I can provide policy details for U.S. states if you specify one.",
                'sources': [] # No relevant sources for unrecognized location
            }

        except Exception as e:
            logger.error(f"Error in get_abortion_policy for '{location_context}': {e}", exc_info=True)
            return self._get_fallback_data(location_context) # Fallback using the provided context



    def get_policy_response(self, question: str, location_context: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Processes a policy-related question and returns structured policy data or a request for location.
        Acts as the main entry point for policy queries needing state context.

        Args:
            question (str): The user's question.
            location_context (Optional[str]): Location context (state name/code) if already known.
            conversation_history (Optional[List[Dict[str, Any]]]): Past conversation messages.

        Returns:
            Dict[str, Any]: A dictionary containing either policy data or an error/request message.
        """
        state_code = None

        # 1. Use provided location_context if available
        if location_context:
            state_code = self.get_state_code(location_context)
            if state_code:
                 logger.info(f"Using provided location context: {location_context} -> {state_code}")

        # 2. If no state code yet, extract from the question
        if not state_code:
            state_code = self._extract_state_from_question(question)
            if state_code:
                logger.info(f"Extracted state code '{state_code}' from question.")

        # 3. If still no state code, check conversation history
        if not state_code and conversation_history:
            # Look back for state mentions or ZIP codes in recent messages
            for msg in reversed(conversation_history[-3:]): # Check last 3 messages
                 content = msg.get('message', '')
                 if content and isinstance(content, str):
                     # Check for state in message content
                     state_from_hist = self._extract_state_from_question(content)
                     if state_from_hist:
                          # Verify it's not ambiguous without context in *this* message
                          is_ambiguous = state_from_hist.lower() in {"in", "me", "or", "hi", "ok", "oh", "la", "pa", "wa", "de", "co", "va", "ma", "id", "mo", "ri"}
                          if is_ambiguous:
                               context_pattern = rf'(?:state|laws|policy|abortion|access|in|for|near)\s+{state_from_hist.lower()}\b|\b{state_from_hist.lower()}\s+(?:state|laws|policy|abortion|access)'
                               if re.search(context_pattern, content.lower()):
                                    state_code = state_from_hist
                                    logger.info(f"Found state code '{state_code}' in conversation history with context.")
                                    break
                               else:
                                    logger.debug(f"Skipping ambiguous state {state_from_hist} from history due to lack of context in that message.")
                          else:
                               state_code = state_from_hist
                               logger.info(f"Found state code '{state_code}' in conversation history.")
                               break

                     # Check for ZIP in history if state not found yet
                     if not state_code:
                          zip_match = re.search(r'\b(\d{5})\b', content)
                          if zip_match:
                               zip_code = zip_match.group(1)
                               # Use fallback ZIP lookup for robustness, can be replaced with better method if available
                               state_from_zip = self._get_state_from_zip_fallback(zip_code)
                               if state_from_zip:
                                    state_code = state_from_zip
                                    logger.info(f"Found state code '{state_code}' from ZIP '{zip_code}' in history.")
                                    break


        # 4. If state code found, get policy data
        if state_code:
            return self.get_policy_data(state_code) # This returns structured data or fallback

        # 5. If no state code found, ask for location
        else:
            logger.info("No state context found for policy question, requesting location information.")
            # Return a structured message indicating location is needed
            return {
                'error': True, # Technically an error state for processing the policy request
                'needs_location': True,
                'message': (
                    "I understand you're asking about abortion policy. To give you the right information, "
                    "could you please let me know which U.S. state or ZIP code you're interested in? "
                    "Laws vary significantly by state."
                ),
                 'sources': [] # No specific sources for this prompt
            }


# --- END OF FILE policy_api.py ---