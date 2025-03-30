import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import sys

# Import your own modules
from weather_data import fetch_weather_data, fetch_forecast_data
from local_preprocessing import preprocess_local_data
from local_prediction import load_model, predict_rainfall

# Import your existing feature engineering functions
sys.path.append('./src')  # Make sure src directory is in path
from feature_engineering import create_features
from advanced_features import (
    create_advanced_weather_features,
    create_weather_event_indicators,
    bin_numerical_features,
    create_polynomial_interactions
)

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

def main():
    # Configuration
    api_key = "80d4ccaf714669a225af28712fccd048"  # Replace with your valid OpenWeatherMap API key
    city_name = "Portland,OR"  # E.g., "London,UK"
    model_path = "../models/catboost_model_20250330_085105.cbm"  # Using the model you specified
    forecast_days = 8  # Number of days to forecast
    
    # Define feature engineering pipeline
    feature_engineering_funcs = [
        create_features,
        create_advanced_weather_features,
        create_weather_event_indicators,
        bin_numerical_features,
        lambda df: create_polynomial_interactions(df, degree=2)
    ]
    
    # Categorical features that need to be treated as such
    categorical_features = [
        "cloud_humidity", "humidity_very_low", "humidity_low", 
        "humidity_moderate", "humidity_high", "cloud_clear",
        "cloud_partly_cloudy", "cloud_mostly_cloudy", "cloud_overcast"
    ]
    
    # 1. Load the model
    try:
        model = load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 2. Get current weather data
    today_weather = fetch_weather_data(api_key, city_name)
    if not today_weather:
        logger.error("Failed to fetch current weather data")
        return
    
    # 3. Preprocess today's weather data
    today_processed = preprocess_local_data(
        today_weather, 
        feature_engineering_funcs,
        categorical_features
    )
    
    # 4. Make prediction for today
    today_prob = predict_rainfall(today_processed, model, categorical_features)
    if today_prob is not None:
        print(f"\n===== Weather Prediction for {city_name} =====")
        print(f"\nTODAY ({datetime.now().strftime('%Y-%m-%d')})")
        print(f"Rainfall probability: {today_prob[0]:.2%}")
        print(f"Recommendation: {'☂️ Bring an umbrella!' if today_prob[0] > 0.5 else '☀️ No umbrella needed.'}")
    
    # 5. Get forecast data for next several days
    forecast_data = fetch_forecast_data(api_key, city_name, days=forecast_days)
    if forecast_data is None:
        logger.error("Failed to fetch forecast data")
        return
    
    print(f"\n8-DAY FORECAST:")
    print("-" * 50)
    
    # Process each day's forecast
    for date, day_forecast in sorted(forecast_data.items()):
        # Skip today (we already handled it above)
        if date == datetime.now().strftime("%Y-%m-%d"):
            continue
            
        # Preprocess this day's forecast data
        day_processed = preprocess_local_data(
            day_forecast, 
            feature_engineering_funcs,
            categorical_features
        )
        
        # Make predictions for this day
        day_probs = predict_rainfall(day_processed, model, categorical_features)
        
        if day_probs is not None:
            # Convert date string to datetime for better formatting
            day_date = datetime.strptime(date, "%Y-%m-%d")
            day_name = day_date.strftime("%A")  # Get day of week name
            
            # Calculate max probability and majority vote
            max_prob = max(day_probs)
            avg_prob = sum(day_probs) / len(day_probs)
            
            # Get some additional weather info for display
            avg_temp = day_forecast['temperature'].mean()
            max_temp = day_forecast['maxtemp'].max()
            min_temp = day_forecast['mintemp'].min()
            
            # Display forecast for this day
            print(f"\n{day_name}, {day_date.strftime('%b %d, %Y')}")
            print(f"Temperature: {avg_temp:.1f}°C (Min: {min_temp:.1f}°C, Max: {max_temp:.1f}°C)")
            print(f"Rainfall probability: {avg_prob:.2%}")
            print(f"Recommendation: {'☂️ Bring an umbrella!' if avg_prob > 0.5 else '☀️ No umbrella needed.'}")
            
    print("\n" + "-" * 50)
    print("Forecast generated using CatBoost model trained on Kaggle rainfall competition data")
    print("Data provided by OpenWeatherMap API")
    print("-" * 50)

if __name__ == "__main__":
    main()