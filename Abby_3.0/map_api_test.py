#!/usr/bin/env python3
import requests
import os
import json
import sys

def test_google_maps_api(api_key=None):
    """Test if the Google Maps API key is working by testing geocoding and places API"""
    
    if not api_key:
        # Try to get from environment variable
        api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
        
    if not api_key:
        # Use the key from the config file
        try:
            from config import GOOGLE_MAPS_API_KEY
            api_key = GOOGLE_MAPS_API_KEY
        except ImportError:
            pass
        
    if not api_key:
        print("❌ ERROR: No Google Maps API key provided.")
        print("Please set the GOOGLE_MAPS_API_KEY environment variable or pass as argument")
        sys.exit(1)
    
    print(f"Testing Google Maps API with key starting with: {api_key[:5]}...")
    
    # Test 1: Geocoding API
    print("\n=== Testing Geocoding API ===")
    geocoding_url = f"https://maps.googleapis.com/maps/api/geocode/json?address=San+Francisco,+CA&key={api_key}"
    
    try:
        response = requests.get(geocoding_url, timeout=10)
        result = response.json()
        
        print(f"Status code: {response.status_code}")
        print(f"API response status: {result.get('status')}")
        
        if response.status_code == 200 and result.get('status') == 'OK':
            location = result['results'][0]['geometry']['location']
            print(f"✅ Geocoding API working! San Francisco coordinates: {location}")
            lat, lng = location['lat'], location['lng']
        else:
            print(f"❌ Geocoding API error: {result.get('error_message', 'Unknown error')}")
            print(f"Full response: {json.dumps(result, indent=2)}")
            return False
    except Exception as e:
        print(f"❌ Error testing Geocoding API: {str(e)}")
        return False
    
    # Test 2: Places API (Nearby Search)
    print("\n=== Testing Places API (Nearby Search) ===")
    places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=5000&type=health&key={api_key}"
    
    try:
        response = requests.get(places_url, timeout=10)
        result = response.json()
        
        print(f"Status code: {response.status_code}")
        print(f"API response status: {result.get('status')}")
        
        if response.status_code == 200 and result.get('status') == 'OK':
            places_count = len(result.get('results', []))
            print(f"✅ Places API working! Found {places_count} places near San Francisco")
            if places_count > 0:
                place = result['results'][0]
                print(f"First place: {place.get('name')} at {place.get('vicinity')}")
        else:
            print(f"❌ Places API error: {result.get('error_message', 'Unknown error')}")
            print(f"Full response: {json.dumps(result, indent=2)}")
            return False
    except Exception as e:
        print(f"❌ Error testing Places API: {str(e)}")
        return False
    
    # Test 3: Places API (Text Search)
    print("\n=== Testing Places API (Text Search) ===")
    text_search_url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=reproductive+health+clinic+san+francisco&key={api_key}"
    
    try:
        response = requests.get(text_search_url, timeout=10)
        result = response.json()
        
        print(f"Status code: {response.status_code}")
        print(f"API response status: {result.get('status')}")
        
        if response.status_code == 200 and result.get('status') == 'OK':
            places_count = len(result.get('results', []))
            print(f"✅ Text Search API working! Found {places_count} clinics in San Francisco")
            if places_count > 0:
                for i, place in enumerate(result['results'][:3]):
                    print(f"{i+1}. {place.get('name')} at {place.get('formatted_address')}")
        else:
            print(f"❌ Text Search API error: {result.get('error_message', 'Unknown error')}")
            print(f"Full response: {json.dumps(result, indent=2)}")
            return False
    except Exception as e:
        print(f"❌ Error testing Text Search API: {str(e)}")
        return False
    
    print("\n✅ All Google Maps API tests passed successfully!")
    return True

if __name__ == "__main__":
    # Get API key from command line if provided
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    test_google_maps_api(api_key) 