#!/usr/bin/env python3
import os
import re
import webbrowser
import platform
import sys

def main():
    # Read .env file to extract Google Maps API key
    api_key = None
    try:
        with open('.env', 'r') as env_file:
            for line in env_file:
                if line.startswith('GOOGLE_MAPS_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
                    # Remove quotes if present
                    api_key = api_key.strip('"\'')
                    break
    except FileNotFoundError:
        print("Error: .env file not found")
        sys.exit(1)
    
    if not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not found in .env file")
        sys.exit(1)
    
    print(f"Found Google Maps API Key: {api_key[:10]}... (length: {len(api_key)})")
    
    # Read the test HTML file
    html_content = None
    try:
        with open('test_maps.html', 'r') as html_file:
            html_content = html_file.read()
    except FileNotFoundError:
        print("Error: test_maps.html file not found")
        sys.exit(1)
    
    # Replace the API key placeholder
    html_content = html_content.replace('let apiKey = \'\';', f'let apiKey = \'{api_key}\';')
    
    # Auto-set the API key and load it immediately
    html_content = html_content.replace(
        'log(\'Test page loaded. Please set your API key and load the Google Maps API.\');', 
        'log(\'Test page loaded with API key. Loading Google Maps API...\');\n'
        '        // Auto-load the Google Maps API\n'
        '        setTimeout(function() {\n'
        '            document.getElementById(\'load-api\').click();\n'
        '        }, 500);'
    )
    
    # Write to a temporary HTML file
    tmp_html = "test_maps_with_key.html"
    with open(tmp_html, 'w') as tmp_file:
        tmp_file.write(html_content)
    
    print(f"Created test file: {tmp_html}")
    
    # Open the HTML file in the default browser
    print("Opening test HTML file in browser...")
    webbrowser.open('file://' + os.path.realpath(tmp_html))
    
    print("Test script is running. Please check your browser.")
    print(f"After testing, you can delete the temporary file: {tmp_html}")

if __name__ == "__main__":
    main() 