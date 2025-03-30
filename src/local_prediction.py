import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import logging
import os
from datetime import datetime

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

def load_model(model_path):
    """Load the trained CatBoost model."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict_rainfall(weather_data, model, categorical_features=None):
    """Predict rainfall probability using the trained model."""
    logger.info("Making rainfall predictions")
    
    try:
        # Convert categorical features to string if needed
        data = weather_data.copy()
        if categorical_features:
            for col in categorical_features:
                if col in data.columns:
                    data[col] = data[col].astype(str)
        
        # Make prediction
        if isinstance(data, pd.DataFrame):
            rainfall_prob = model.predict_proba(data)[:, 1]
            logger.info(f"Successfully made predictions for {len(data)} samples")
            return rainfall_prob
        else:
            logger.error("Input data is not a DataFrame")
            return None
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None