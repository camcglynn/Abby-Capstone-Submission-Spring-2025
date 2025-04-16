import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:5006"  # Updated to correct port
TEST_CASES = [
    {
        "name": "Test 1: ZIP Code Detection", 
        "message": "Where can I find an abortion clinic in 95123?",
        "expected": {"is_location_clinic_query": True, "zip_code": "95123"}
    },
    {
        "name": "Test 2: City Name Detection", 
        "message": "Are there abortion clinics near San Jose?",
        "expected": {"is_location_clinic_query": True, "city_name": "San Jose"}
    },
    {
        "name": "Test 3: No Location", 
        "message": "I need to find an abortion clinic",
        "expected": {"is_location_clinic_query": True}
    },
    {
        "name": "Test 4: Non-clinic query", 
        "message": "What are the symptoms of pregnancy?",
        "expected": {"is_location_clinic_query": False}
    }
]

def run_test_case(test_case):
    """Run a single test case against the API"""
    print(f"\n===== {test_case['name']} =====")
    print(f"Query: '{test_case['message']}'")
    
    # First test the local detection function
    result = detect_location_based_clinic_query(test_case['message'])
    print(f"Local detection: {json.dumps(result, indent=2)}")
    
    # Then test against the server
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": test_case['message'],
                "session_id": "test_session_123"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Server response status: Success (200)")
            
            # Check for map-related fields in response
            show_map = response_data.get("show_map", False)
            zip_code = response_data.get("zip_code")
            
            print(f"show_map: {show_map}")
            print(f"zip_code/location: {zip_code or 'None'}")
            
            # Basic validation
            if test_case['expected']['is_location_clinic_query'] and not show_map:
                print("\033[91m✗ FAIL\033[0m: Expected show_map=True but got False")
            elif not test_case['expected']['is_location_clinic_query'] and show_map:
                print("\033[91m✗ FAIL\033[0m: Expected show_map=False but got True")
            elif test_case['expected'].get('zip_code') and test_case['expected']['zip_code'] != zip_code:
                print(f"\033[91m✗ FAIL\033[0m: Expected zip_code={test_case['expected']['zip_code']} but got {zip_code}")
            else:
                print("\033[92m✓ PASS\033[0m: Detection and map display match expectations")
        else:
            print(f"\033[91mFAILED\033[0m: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"\033[91mERROR\033[0m: {str(e)}")

def detect_location_based_clinic_query(message):
    """Implementation of the detection function from chat.js"""
    # Check if the message is about finding abortion clinics
    clinic_keywords = ['abortion clinic', 'abortion provider', 'planned parenthood', 'family planning', 'women\'s health']
    location_keywords = ['near', 'in', 'around', 'close to', 'nearby']
    
    # Convert message to lowercase for case-insensitive matching
    lowercase_message = message.lower()
    
    # Check if message contains clinic keywords
    has_clinic_keyword = any(keyword in lowercase_message for keyword in clinic_keywords)
    
    # Check if message contains location keywords
    has_location_keyword = any(keyword in lowercase_message for keyword in location_keywords)
    
    # Check for ZIP code using regex (5-digit US ZIP code)
    import re
    zip_code_match = re.search(r'\b\d{5}\b', message)
    zip_code = zip_code_match.group(0) if zip_code_match else None
    
    # Check for city name using regex
    city_match = None
    for keyword in location_keywords:
        pattern = fr'{keyword}\s+([A-Za-z\s]+)(?:[,.]|$)'
        match = re.search(pattern, message, re.IGNORECASE)
        if match and match.group(1):
            city_match = match.group(1).strip()
            break
    
    # Return result
    return {
        "isLocationClinicQuery": has_clinic_keyword and (has_location_keyword or zip_code),
        "zipCode": zip_code,
        "cityName": city_match
    }

def main():
    """Run all test cases"""
    print("=== Testing Location-Based Clinic Queries ===\n")
    
    # Check if the server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"\033[91mERROR\033[0m: Server not running at {BASE_URL}")
        print("Please start the server before running this test")
        sys.exit(1)
    
    for test_case in TEST_CASES:
        run_test_case(test_case)
    
    print("\n=== Testing complete ===")

if __name__ == "__main__":
    main() 