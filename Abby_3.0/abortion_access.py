"""
Utility module for abortion access information by location
"""

# States where abortion is legal
ABORTION_LEGAL_STATES = [
    "Alaska", "California", "Colorado", "Connecticut", "Delaware", 
    "Hawaii", "Illinois", "Maine", "Maryland", "Massachusetts", 
    "Michigan", "Minnesota", "Montana", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "Oregon", "Pennsylvania", 
    "Rhode Island", "Vermont", "Virginia", "Washington"
]

# Lookup for nearby less restrictive states
NEARBY_SIGNIFICANTLY_LESS_RESTRICTIVE_STATES = {
    "Alabama": [],
    "Arkansas": ["Kansas", "Illinois"], 
    "Idaho": ["Washington", "Oregon", "Nevada", "Montana"],
    "Kentucky": ["Illinois", "Virginia"],
    "Louisiana": [],
    "Mississippi": [],
    "Missouri": ["Illinois", "Kansas"],
    "North Dakota": ["Minnesota", "Montana"],
    "Oklahoma": ["Kansas", "Colorado", "New Mexico"],
    "South Dakota": ["Minnesota", "Montana"],
    "Tennessee": ["Virginia", "Illinois"], 
    "Texas": ["New Mexico"],
    "West Virginia": ["Pennsylvania", "Maryland", "Virginia"],
    "Wisconsin": ["Illinois", "Minnesota", "Michigan"],
    "Arizona": ["California", "Nevada", "Colorado", "New Mexico"],
    "Florida": [],
    "Georgia": ["North Carolina", "Florida"], 
    "Indiana": ["Illinois", "Michigan"],
    "Iowa": ["Illinois", "Minnesota", "Nebraska"],
    "Nebraska": ["Kansas", "Colorado"],
    "North Carolina": ["Virginia"],
    "Ohio": ["Pennsylvania", "Michigan"],
    "South Carolina": ["North Carolina"], 
    "Utah": ["Colorado", "Nevada", "New Mexico"]
}

# Policy details by state
STATE_POLICY_DETAILS = {
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
    # Add more states as needed
}

# Full ZIP code to state mapping
from .zip_codes import ZIP_TO_STATE

def get_state_from_zip(zip_code):
    """Get state name from a ZIP code"""
    if not zip_code or len(zip_code) < 3:
        return None
        
    prefix = zip_code[:3]
    return ZIP_TO_STATE.get(prefix)

def is_abortion_legal_in_state(state_name):
    """Check if abortion is legal in a given state"""
    if not state_name:
        return None
    
    # Normalize state name for comparison
    normalized_state = state_name.strip().title()
    
    return normalized_state in ABORTION_LEGAL_STATES

def get_policy_for_state(state_name):
    """Get abortion policy details for a given state"""
    if not state_name:
        return None
    
    # Normalize state name for comparison
    normalized_state = state_name.strip().title()
    
    # Try to get from detailed policy lookup
    policy = STATE_POLICY_DETAILS.get(normalized_state)
    
    # If not in detailed lookup, create basic policy based on legal status
    if not policy:
        is_legal = normalized_state in ABORTION_LEGAL_STATES
        policy = {
            "legal": is_legal,
            "summary": f"Abortion is {'legal' if is_legal else 'not legal'} in {normalized_state}.",
            "gestational_limit": "Policy details not available" if is_legal else "Abortion is banned with limited exceptions.",
            "insurance_coverage": "Policy details not available",
            "waiting_period": "Policy details not available"
        }
    
    return policy

def get_less_restrictive_states(state_name):
    """Get nearby less restrictive states for a given state"""
    if not state_name:
        return []
    
    # Normalize state name for comparison
    normalized_state = state_name.strip().title()
    
    return NEARBY_SIGNIFICANTLY_LESS_RESTRICTIVE_STATES.get(normalized_state, []) 