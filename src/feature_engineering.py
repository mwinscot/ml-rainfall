import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_features(df):
    """Create new features from existing weather data."""
    logger.info("Creating new features")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Check if we have the necessary columns
    if all(col in df.columns for col in ['humidity', 'temperature', 'pressure']):
        # Temperature-humidity interaction (heat index approximation)
        df_new['temp_humidity'] = df_new['temperature'] * df_new['humidity']
        
        # Dew point and temperature difference (important for precipitation)
        if 'dewpoint' in df.columns:
            df_new['temp_dewpoint_diff'] = df_new['temperature'] - df_new['dewpoint']
        
        # Pressure gradient (rapid pressure changes often lead to weather changes)
        # This would be better with time series data, but we can make a feature for model training
        if 'pressure' in df.columns:
            df_new['pressure_scaled'] = df_new['pressure'] / 1013.25  # Normalize to standard pressure
    
    # Wind related features
    if all(col in df.columns for col in ['windspeed', 'winddirection']):
        # Convert wind direction to radians for trigonometric functions
        if 'winddirection' in df.columns:
            # Handle potential missing values
            df_new['winddirection'] = df_new['winddirection'].fillna(df_new['winddirection'].median())
            
            # Convert to radians
            wind_rad = np.radians(df_new['winddirection'])
            
            # Wind components
            df_new['wind_x'] = df_new['windspeed'] * np.cos(wind_rad)
            df_new['wind_y'] = df_new['windspeed'] * np.sin(wind_rad)
    
    # Temperature range related
    if all(col in df.columns for col in ['maxtemp', 'mintemp']):
        df_new['temp_range'] = df_new['maxtemp'] - df_new['mintemp']
    
    # Cloud cover and humidity interaction
    if all(col in df.columns for col in ['cloud', 'humidity']):
        df_new['cloud_humidity'] = df_new['cloud'] * df_new['humidity']
    
    # Log the new features created
    new_features = [col for col in df_new.columns if col not in df.columns]
    logger.info(f"Created {len(new_features)} new features: {new_features}")
    
    return df_new