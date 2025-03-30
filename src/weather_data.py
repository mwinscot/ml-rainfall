import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("local_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_coordinates(api_key, city_name):
    """Get lat and lon coordinates for a city name."""
    logger.info(f"Getting coordinates for {city_name}")
    
    # Use OpenWeatherMap Geocoding API to get coordinates
    base_url = "http://api.openweathermap.org/geo/1.0/direct?"
    complete_url = f"{base_url}q={city_name}&limit=1&appid={api_key}"
    
    try:
        response = requests.get(complete_url)
        data = response.json()
        
        if data and len(data) > 0:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
            logger.info(f"Got coordinates for {city_name}: lat={lat}, lon={lon}")
            return lat, lon
        else:
            logger.error(f"Could not find coordinates for {city_name}")
            return None, None
    
    except Exception as e:
        logger.error(f"Error getting coordinates: {e}")
        return None, None

def fetch_onecall_data(api_key, city_name):
    """Fetch current and forecast weather data using OneCall API."""
    logger.info(f"Fetching OneCall weather data for {city_name}")
    
    # First get coordinates
    lat, lon = get_coordinates(api_key, city_name)
    if lat is None or lon is None:
        return None, None
    
    # Use OneCall API to get current and forecast data
    base_url = "https://api.openweathermap.org/data/3.0/onecall?"
    complete_url = f"{base_url}lat={lat}&lon={lon}&exclude=minutely,alerts&units=metric&appid={api_key}"
    
    try:
        response = requests.get(complete_url)
        data = response.json()
        
        # Log the response for debugging
        logger.info(f"API Response status code: {response.status_code}")
        logger.debug(f"API Response: {data}")
        
        if response.status_code == 200:
            # Process current weather
            current_weather = process_current_data(data["current"])
            
            # Process daily forecasts
            tomorrow_forecast = process_forecast_data(data["hourly"], data["daily"])
            
            logger.info(f"Successfully fetched OneCall data for {city_name}")
            return current_weather, tomorrow_forecast
        else:
            logger.error(f"Error from API: {data.get('message', 'Unknown error')}")
            return None, None
    
    except Exception as e:
        logger.error(f"Error fetching OneCall data: {e}")
        return None, None

def process_current_data(current_data):
    """Process current weather data from OneCall API."""
    try:
        # Calculate dewpoint
        temp = current_data["temp"]
        humidity = current_data["humidity"]
        
        # Constants for Magnus formula
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity/100.0)
        dewpoint = (b * alpha) / (a - alpha)
        
        # Get sunshine hours (approximate based on cloud cover)
        sunshine = 1 * (1 - current_data.get("clouds", 0) / 100)  # 1 hour * clear sky percentage
        
        weather_dict = {
            "temperature": temp,
            "pressure": current_data.get("pressure", 1013),
            "humidity": humidity,
            "maxtemp": temp + 2,  # approximation
            "mintemp": temp - 2,  # approximation
            "dewpoint": dewpoint,
            "windspeed": current_data.get("wind_speed", 0),
            "winddirection": current_data.get("wind_deg", 0),
            "cloud": current_data.get("clouds", 0),
            "sunshine": sunshine,
            "day": datetime.fromtimestamp(current_data["dt"]).day
        }
        
        return weather_dict
    
    except Exception as e:
        logger.error(f"Error processing current data: {e}")
        return None

def process_forecast_data(hourly_data, daily_data):
    """Process forecast data from OneCall API for all available days."""
    try:
        forecasts_by_day = {}
        
        # Get hourly forecasts first (more detailed)
        for hour in hourly_data:
            forecast_time = datetime.fromtimestamp(hour["dt"])
            forecast_date = forecast_time.strftime("%Y-%m-%d")
            
            # Calculate dewpoint
            temp = hour["temp"]
            humidity = hour["humidity"]
            
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(humidity/100.0)
            dewpoint = (b * alpha) / (a - alpha)
            
            # Calculate sunshine (daytime hours only)
            sunshine = 0
            if 6 <= forecast_time.hour <= 18:  # Daytime hours
                sunshine = 1 * (1 - hour.get("clouds", 0) / 100)  # 1 hour * clear sky percentage
            
            forecast_dict = {
                "datetime": forecast_time.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": temp,
                "pressure": hour.get("pressure", 1013),
                "humidity": humidity,
                "maxtemp": temp + 1,  # approximation from hourly data
                "mintemp": temp - 1,  # approximation from hourly data
                "dewpoint": dewpoint,
                "windspeed": hour.get("wind_speed", 0),
                "winddirection": hour.get("wind_deg", 0),
                "cloud": hour.get("clouds", 0),
                "sunshine": sunshine,
                "day": forecast_time.day,
                "precipitation_prob": hour.get("pop", 0)
            }
            
            # Add to the forecasts for this day
            if forecast_date not in forecasts_by_day:
                forecasts_by_day[forecast_date] = []
            forecasts_by_day[forecast_date].append(forecast_dict)
        
        # Extend with daily data for days beyond the hourly forecast
        for day in daily_data:
            day_date = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
            
            # Skip if we already have hourly data for this day
            if day_date in forecasts_by_day and forecasts_by_day[day_date]:
                continue
                
            # Get data for full day
            forecast_time = datetime.fromtimestamp(day["dt"])
            
            # Calculate dewpoint from day temperature and humidity
            temp = day["temp"]["day"]
            humidity = day["humidity"]
            
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(humidity/100.0)
            dewpoint = (b * alpha) / (a - alpha)
            
            # For daily data, estimate sunshine based on cloud cover and uvi
            sunshine = 12 * (1 - day.get("clouds", 0) / 100)  # Assume 12 hours of potential daylight
            
            # Create a single entry for the day
            forecast_dict = {
                "datetime": forecast_time.strftime("%Y-%m-%d 12:00:00"),  # Use noon as representative time
                "temperature": temp,
                "pressure": day.get("pressure", 1013),
                "humidity": humidity,
                "maxtemp": day["temp"]["max"],
                "mintemp": day["temp"]["min"],
                "dewpoint": dewpoint,
                "windspeed": day.get("wind_speed", 0),
                "winddirection": day.get("wind_deg", 0),
                "cloud": day.get("clouds", 0),
                "sunshine": sunshine,
                "day": forecast_time.day,
                "precipitation_prob": day.get("pop", 0)
            }
            
            forecasts_by_day[day_date] = [forecast_dict]
        
        # Create a list of dataframes, one for each day
        forecast_dfs = {}
        for date, forecasts in forecasts_by_day.items():
            forecast_dfs[date] = pd.DataFrame(forecasts)
            
        return forecast_dfs
    
    except Exception as e:
        logger.error(f"Error processing forecast data: {e}")
        return None

# Maintain compatibility with existing code by keeping these functions
def fetch_weather_data(api_key, city_name):
    """Compatibility function for existing code."""
    current, _ = fetch_onecall_data(api_key, city_name)
    return current

def fetch_forecast_data(api_key, city_name, days=8):
    """Compatibility function for existing code.
    Returns forecast data for specified number of days (default: 8)."""
    _, forecasts = fetch_onecall_data(api_key, city_name)
    if forecasts is None:
        return None
        
    # Get today's date to filter forecasts
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Convert to list of dataframes sorted by date
    if isinstance(forecasts, dict):
        # Filter for requested number of days
        dates = sorted(list(forecasts.keys()))
        selected_dates = dates[:min(days+1, len(dates))]  # +1 to include today
        
        # If only requesting a single day forecast, return just that dataframe
        if days == 1 and len(selected_dates) > 1:
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            if tomorrow in forecasts:
                return forecasts[tomorrow]
            elif len(selected_dates) > 1:
                return forecasts[selected_dates[1]]  # Return the first day after today
        
        # Return dict of selected days
        return {date: forecasts[date] for date in selected_dates if date in forecasts}