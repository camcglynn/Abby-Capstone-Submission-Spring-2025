"""
Location handler for abortion access information
This module handles interpreting user location inputs and providing abortion access details
"""

import re
from typing import Dict, List, Optional, Any, Union
from .abortion_access import (
    get_state_from_zip,
    is_abortion_legal_in_state,
    get_policy_for_state,
    get_less_restrictive_states
)

# Define ambiguous state abbreviations that should not be detected in lowercase
AMBIGUOUS_ABBREVIATIONS = {"in", "on", "at", "me", "hi", "ok", "or", "la", "pa", "no", "so"}

# States where abortion is legal (at least in some circumstances)
ABORTION_LEGAL_STATES = [
    "Alaska", "California", "Colorado", "Connecticut", "Delaware", 
    "Hawaii", "Illinois", "Maine", "Maryland", "Massachusetts", 
    "Michigan", "Minnesota", "Montana", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "Oregon", "Rhode Island", 
    "Vermont", "Virginia", "Washington", "Wyoming", "District of Columbia"
]

# Mapping of states to nearby states with less restrictive abortion policies
NEARBY_LESS_RESTRICTIVE_STATES = {
    # States with Total or Near-Total Bans
    "Alabama": ["Florida", "Georgia"], # FL/GA have restrictions, but less than AL's ban
    "Arkansas": ["Kansas", "Illinois", "Missouri"], # MO has a ban, but IL/KS are less restrictive. TN/MS/LA/OK/TX have bans/restrictions.
    "Idaho": ["Washington", "Oregon", "Nevada", "Montana"], # UT/WY have restrictions/bans
    "Kentucky": ["Illinois", "Virginia", "Ohio"], # IN/WV/TN/MO have restrictions/bans. Ohio's status is complex but generally less restrictive than KY's ban.
    "Louisiana": [], # Borders states with bans (TX, AR, MS). Florida is closest non-bordering w/ restrictions.
    "Mississippi": ["Tennessee"], # Borders states with bans (LA, AR, AL). TN has a ban but exceptions might differ slightly? Closest less restrictive are FL/GA/IL. 
    "Missouri": ["Illinois", "Kansas", "Iowa", "Nebraska"], # KY/TN/AR/OK have bans/restrictions. IA/NE have restrictions but less than MO's ban.
    "North Dakota": ["Minnesota", "Montana"], # SD has ban.
    "Oklahoma": ["Kansas", "Colorado", "New Mexico"], # MO/AR/TX have bans/restrictions.
    "South Dakota": ["Minnesota", "Montana", "Iowa", "Nebraska"], # ND/WY have bans/restrictions. IA/NE have restrictions but less than SD's ban.
    "Tennessee": ["Virginia", "North Carolina", "Georgia", "Illinois", "Missouri", "Kentucky"], # AL/MS/AR have bans. NC/GA/KY/MO have restrictions/bans but potentially less absolute than TN. VA/IL are less restrictive.
    "Texas": ["New Mexico", "Oklahoma", "Arkansas", "Louisiana"], # OK/AR/LA have bans/restrictions but NM is less restrictive.
    "West Virginia": ["Pennsylvania", "Maryland", "Virginia", "Ohio"], # KY has ban. Ohio status complex but less restrictive.
    "Wisconsin": ["Illinois", "Minnesota", "Michigan", "Iowa"], # Iowa has restrictions but less than WI's uncertain/potentially banned status.

    # States with Severe Gestational Limits (e.g., 6-15 weeks)
    "Arizona": ["California", "Nevada", "Colorado", "New Mexico"], # Utah has restrictions.
    "Florida": ["Georgia"], # AL has ban. GA has restrictions but is bordering.
    "Georgia": ["Florida", "North Carolina", "South Carolina", "Tennessee"], # AL/TN have bans. FL/NC/SC have restrictions but differ from GA's.
    "Indiana": ["Illinois", "Michigan", "Ohio"], # KY has ban. Ohio status complex but less restrictive.
    "Iowa": ["Illinois", "Minnesota", "Nebraska", "Wisconsin", "Missouri"], # SD/MO have bans/restrictions. WI status uncertain. NE less restrictive.
    "Nebraska": ["Kansas", "Colorado", "Iowa", "Missouri", "Wyoming"], # SD/WY have bans/restrictions. MO has ban. Iowa less restrictive.
    "North Carolina": ["Virginia", "Tennessee", "South Carolina", "Georgia"], # TN has ban. VA less restrictive. SC/GA have restrictions.
    "Ohio": ["Pennsylvania", "Michigan", "Indiana", "Kentucky", "West Virginia"], # KY/WV/IN have bans/restrictions. PA/MI less restrictive.
    "South Carolina": ["North Carolina", "Georgia"], # Both neighbors have restrictions.
    "Utah": ["Colorado", "Nevada", "Arizona", "New Mexico", "Idaho", "Wyoming"] # ID/WY have bans/restrictions. CO/NV/AZ/NM less restrictive.
}

# Filtered version focusing only on significantly less restrictive bordering states
NEARBY_SIGNIFICANTLY_LESS_RESTRICTIVE_STATES = {
    "Alabama": [],
    "Arkansas": ["Kansas", "Illinois"], # Via MO border proximity for IL
    "Idaho": ["Washington", "Oregon", "Nevada", "Montana"],
    "Kentucky": ["Illinois", "Virginia"],
    "Louisiana": [],
    "Mississippi": [],
    "Missouri": ["Illinois", "Kansas"],
    "North Dakota": ["Minnesota", "Montana"],
    "Oklahoma": ["Kansas", "Colorado", "New Mexico"],
    "South Dakota": ["Minnesota", "Montana"],
    "Tennessee": ["Virginia", "Illinois"], # Via KY/MO borders
    "Texas": ["New Mexico"],
    "West Virginia": ["Pennsylvania", "Maryland", "Virginia"],
    "Wisconsin": ["Illinois", "Minnesota", "Michigan"],
    "Arizona": ["California", "Nevada", "Colorado", "New Mexico"],
    "Florida": [],
    "Georgia": ["North Carolina", "Florida"], # NC/FL are less restrictive than 6 weeks
    "Indiana": ["Illinois", "Michigan"],
    "Iowa": ["Illinois", "Minnesota", "Nebraska"],
    "Nebraska": ["Kansas", "Colorado"],
    "North Carolina": ["Virginia"],
    "Ohio": ["Pennsylvania", "Michigan"],
    "South Carolina": ["North Carolina"], # NC less restrictive than SC
    "Utah": ["Colorado", "Nevada", "New Mexico"]
}

def extract_location_from_input(input_text: str) -> Optional[str]:
    """Extract location (state name, abbreviation, or ZIP code) from input text"""
    if not input_text:
        return None
        
    # Check for ZIP code first (most reliable)
    zip_match = re.search(r'\b(\d{5})\b', input_text)
    if zip_match:
        return zip_match.group(1)
    
    # Check for full state names
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
    
    # Check for full state names (most reliable)
    for state_name, abbr in state_names.items():
        if re.search(r'\b' + re.escape(state_name) + r'\b', input_text, re.IGNORECASE):
            return state_name
    
    # Check for state abbreviations, being careful with ambiguous ones
    for abbr, state_name in state_names.items():
        abbr_lower = abbr.lower()
        
        # Handle ambiguous abbreviations carefully
        if abbr_lower in AMBIGUOUS_ABBREVIATIONS:
            # Only match ambiguous abbreviations in uppercase or clear location context
            if abbr in input_text:  # Exact case match (uppercase)
                return state_name
            
            # Check for clear location context patterns
            location_patterns = [
                rf'state:?\s*{re.escape(abbr)}',  # "State: ME"
                rf'(?:in|from|to|at|live(?:s|d)? in|move(?:d|ing)? to|reside(?:s|d)? in)\s+{re.escape(abbr)}(?:\b|\s|$)',
                rf'(?:abortion|laws|policy|policies) (?:in|for) {re.escape(abbr)}',
                rf'{re.escape(abbr)} state',
                rf'state of {re.escape(abbr)}'
            ]
            
            if any(re.search(pattern, input_text, re.IGNORECASE) for pattern in location_patterns):
                return state_name
        else:
            # For non-ambiguous state codes, match normally
            if f" {abbr} " in f" {input_text} " or f" {abbr}," in f" {input_text} " or f" {abbr}." in f" {input_text} ":
                return state_name
    
    return None

def get_abortion_access_info(location_input):
    """Get information about abortion access for a given location"""
    result = {}
    
    # Extract location from input
    location = extract_location_from_input(location_input)
    if not location:
        return {
            "error": "I couldn't determine which state you're asking about. Please specify a U.S. state or ZIP code so I can provide accurate information about abortion access in your area."
        }
    
    # Get state name from location
    state_name = None
    
    # Check if it's a ZIP code
    if isinstance(location, str) and location.isdigit() and len(location) == 5:
        state_name = get_state_from_zip(location)
        if not state_name:
            return {
                "error": f"I couldn't determine which state the ZIP code {location} belongs to. Please provide a state name instead."
            }
    else:
        # Assume it's a state name - normalize it
        state_name = location.strip().title()
    
    result["state"] = state_name
    
    # Check if abortion is legal in the state
    is_legal = state_name in ABORTION_LEGAL_STATES
    result["is_legal"] = is_legal
    
    # Get policy details for the state
    policy = get_policy_for_state(state_name)
    result["policy"] = policy
    
    # Get nearby less restrictive states for travel options
    if not is_legal:
        # Use significantly less restrictive states first
        nearby_states = NEARBY_SIGNIFICANTLY_LESS_RESTRICTIVE_STATES.get(state_name, [])
        
        # If no significantly less restrictive states, use all less restrictive states
        if not nearby_states:
            nearby_states = NEARBY_LESS_RESTRICTIVE_STATES.get(state_name, [])
        
        result["nearby_less_restrictive_states"] = nearby_states
    
    return result

def generate_abortion_access_response(location_input):
    """Generate a user-friendly response about abortion access based on location input"""
    info = get_abortion_access_info(location_input)
    
    # Handle error
    if "error" in info:
        return {
            "text": info["error"],
            "error": True
        }
    
    # Build response based on legal status
    state_name = info["state"]
    is_legal = info["is_legal"]
    policy = info["policy"]
    
    if is_legal:
        # Case 1: Abortion is legal in the state
        response_text = f"Yes, abortion is legal in {state_name}.\n\n"
        
        # Add policy details
        response_text += f"• {policy['gestational_limit']}\n"
        response_text += f"• {policy['insurance_coverage']}\n"
        response_text += f"• {policy['waiting_period']}\n\n"
        response_text += f"{policy['summary']}\n\n"
        response_text += "You can see nearby clinics on the map below."
        
        return {
            "text": response_text,
            "show_map": True,
            "map_query": f"abortion clinic in {state_name}",
            "state_name": state_name,
            "is_legal": True
        }
    else:
        # Case 2: Abortion is not legal in the state
        response_text = f"Abortion is not currently legal in {state_name}.\n\n"
        
        # Add policy details but more limited since it's banned
        response_text += f"• {policy['gestational_limit']}\n"
        
        # Add nearby less restrictive states information
        nearby_states = info["nearby_less_restrictive_states"]
        if nearby_states:
            response_text += f"\nHere are nearby states with less restrictive abortion laws where you may be able to access care:\n"
            for state in nearby_states:
                response_text += f"• {state}\n"
            
            # Get the first (closest/best) alternative state
            travel_state = nearby_states[0]
            
            # Get that state's policy instead of the banned state
            travel_state_policy = get_policy_for_state(travel_state)
            
            # Add travel suggestions and policy for the alternative state
            response_text += f"\nIn {travel_state}:\n"
            response_text += f"• {travel_state_policy['gestational_limit']}\n"
            response_text += f"• {travel_state_policy['insurance_coverage']}\n"
            response_text += f"• {travel_state_policy['waiting_period']}\n\n"
            
            response_text += f"Traveling for abortion care can be challenging. The map below shows abortion clinics in {travel_state}, and these resources may be able to help with travel and funding:\n"
            response_text += "• National Abortion Federation Hotline: 1-800-772-9100\n"
            response_text += "• Abortion Finder: https://www.abortionfinder.org\n"
            response_text += "• INeedAnA.com: https://www.ineedana.com\n"
            
            return {
                "text": response_text,
                "show_map": True,
                "map_query": f"abortion clinic in {travel_state}",
                "state_name": state_name,
                "is_legal": False,
                "travel_state": travel_state,
                "travel_map": True,  # Add flag for travel map
                "from_state": state_name,
                "to_state": travel_state
            }
        else:
            # No nearby less restrictive states
            response_text += "\nUnfortunately, there are no nearby states with less restrictive abortion laws. You may need to travel further for care or explore telehealth options if available.\n\n"
            response_text += "These resources may be able to help you find care in other regions:\n"
            response_text += "• National Abortion Federation Hotline: 1-800-772-9100\n"
            response_text += "• Abortion Finder: https://www.abortionfinder.org\n"
            response_text += "• INeedAnA.com: https://www.ineedana.com\n"
            
            return {
                "text": response_text,
                "show_map": False,  # Don't show map when there's no good alternative
                "state_name": state_name,
                "is_legal": False
            }
        
        # This return will never be reached due to the if/else branches above
        return {
            "text": response_text,
            "show_map": False,
            "state_name": state_name,
            "is_legal": False
        } 