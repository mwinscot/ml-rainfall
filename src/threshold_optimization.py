import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("threshold_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def optimize_threshold():
    """Create submissions with different probability thresholds."""
    logger.info("Starting threshold optimization")
    
    # Load the best model's predictions
    best_model_file = "../submissions/advanced_catboost_20250330_082904.csv"
    logger.info(f"Loading predictions from {best_model_file}")
    predictions = pd.read_csv(best_model_file)
    
    # Thresholds to try
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    
    # Create submissions for each threshold
    for threshold in thresholds:
        logger.info(f"Processing threshold {threshold}")
        
        # Create a copy of the predictions
        threshold_preds = predictions.copy()
        
        # Apply the threshold
        original_probs = threshold_preds['target'].copy()
        
        # Apply threshold effects - this will increase confidence in predictions
        # above the threshold and decrease confidence in predictions below it
        threshold_preds['target'] = np.where(
            original_probs >= threshold,
            original_probs + (1 - original_probs) * 0.1,  # Boost high probs
            original_probs * 0.9  # Reduce low probs
        )
        
        # Create filename
        threshold_str = str(threshold).replace('.', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../submissions/threshold_{threshold_str}_{timestamp}.csv"
        
        # Save the submission
        threshold_preds.to_csv(filename, index=False)
        logger.info(f"Saved threshold-adjusted predictions to {filename}")
        
        print(f"Created submission with threshold {threshold}: {filename}")
        print(f"To submit to Kaggle, use this command:")
        print(f"kaggle competitions submit -c playground-series-s5e3 -f {filename} -m \"Advanced CatBoost with threshold {threshold}\"")
        print()
    
    logger.info("Threshold optimization complete")
    return

if __name__ == "__main__":
    optimize_threshold()