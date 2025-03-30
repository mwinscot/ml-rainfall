import requests
import json
import os
from datetime import datetime

def test_api_call():
    """Test the OpenWeatherMap OneCall API v3.0"""
    # Get API key from environment variable or config file
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    
    # If environment variable is not set, try to load from config file
    if not api_key:
        try:
            with open('.env.local', 'r') as f:
                for line in f:
                    if line.startswith('OPENWEATHER_API_KEY='):
                        api_key = line.strip().split('=')[1].strip('"\'')
                        break
        except FileNotFoundError:
            print("WARNING: API key not found. Please set OPENWEATHER_API_KEY environment variable")
            print("or create a .env.local file with OPENWEATHER_API_KEY=your_key")
            return
    
    if not api_key:
        print("ERROR: No API key provided. Exiting.")
        return
    
    # Portland, OR coordinates
    lat = 45.5234
    lon = -122.6762
    
    # Test geocoding API first
    print("Testing Geocoding API...")
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q=Portland,OR&limit=1&appid={api_key}"
    geo_response = requests.get(geo_url)
    print(f"Status Code: {geo_response.status_code}")
    
    if geo_response.status_code == 200:
        geo_data = geo_response.json()
        print(json.dumps(geo_data, indent=2))
        if geo_data and len(geo_data) > 0:
            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]
            print(f"Got coordinates: lat={lat}, lon={lon}")
    else:
        print(f"Geocoding API Error: {geo_response.text}")
    
    # Now test the OneCall API
    print("\nTesting OneCall API v3.0...")
    onecall_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&units=metric&appid={api_key}"
    onecall_response = requests.get(onecall_url)
    print(f"Status Code: {onecall_response.status_code}")
    
    if onecall_response.status_code == 200:
        onecall_data = onecall_response.json()
        
        # Print a summary to avoid overwhelming output
        print("\nAPI Call Successful! Summary of data:")
        print(f"Current Temperature: {onecall_data['current']['temp']}°C")
        print(f"Current Weather: {onecall_data['current']['weather'][0]['description']}")
        print(f"Number of hourly forecasts: {len(onecall_data['hourly'])}")
        print(f"Number of daily forecasts: {len(onecall_data['daily'])}")
        
        # Print first daily forecast as sample
        first_day = onecall_data['daily'][0]
        date = datetime.fromtimestamp(first_day['dt']).strftime('%Y-%m-%d')
        print(f"\nSample forecast for {date}:")
        print(f"Temperature: {first_day['temp']['day']}°C (min: {first_day['temp']['min']}°C, max: {first_day['temp']['max']}°C)")
        print(f"Weather: {first_day['weather'][0]['description']}")
        print(f"Precipitation probability: {first_day.get('pop', 0) * 100}%")
        
        # Save full response to a file for detailed inspection
        with open('api_response.json', 'w') as f:
            json.dump(onecall_data, f, indent=2)
        print("\nFull API response saved to api_response.json")
    else:
        print(f"OneCall API Error: {onecall_response.text}")

if __name__ == "__main__":
    test_api_call()