import pandas as pd
import numpy as np
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

def preprocess_local_data(weather_data, feature_engineering_functions, categorical_features=None):
    """Transform raw weather data to match model's expected features."""
    logger.info("Preprocessing local weather data")
    
    # Convert dictionary to DataFrame for a single day
    if isinstance(weather_data, dict):
        df = pd.DataFrame([weather_data])
    else:
        df = weather_data.copy()
    
    # Apply all the same feature engineering steps from your model
    for func in feature_engineering_functions:
        try:
            df = func(df)
            logger.info(f"Applied feature engineering function: {func.__name__}")
        except Exception as e:
            logger.error(f"Error applying feature engineering function {func.__name__}: {e}")
    
    # Handle categorical features
    if categorical_features:
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                logger.info(f"Converted {col} to string type")
    
    # Check for missing columns that were in the training data
    logger.info(f"Final preprocessed data shape: {df.shape}")
    return df