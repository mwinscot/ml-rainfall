import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("super_blend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_super_blend():
    """Create a blend of the advanced CatBoost model and the previous best blend."""
    logger.info("Creating super blend")
    
    # Specify the files to blend
    advanced_catboost_file = "../submissions/advanced_catboost_20250330_082904.csv"
    best_blend_file = "../submissions/blend_060_040_20250329_102222.csv"
    
    # Load the prediction files
    logger.info(f"Loading {advanced_catboost_file}")
    advanced_catboost = pd.read_csv(advanced_catboost_file)
    
    logger.info(f"Loading {best_blend_file}")
    best_blend = pd.read_csv(best_blend_file)
    
    # Verify IDs match
    if not all(advanced_catboost['id'] == best_blend['id']):
        logger.error("IDs do not match between files")
        raise ValueError("IDs do not match between files")
    
    # Set weights
    advanced_catboost_weight = 0.8
    best_blend_weight = 0.2
    
    # Create the super blend
    super_blend = advanced_catboost.copy()
    super_blend['target'] = (
        advanced_catboost_weight * advanced_catboost['target'] +
        best_blend_weight * best_blend['target']
    )
    
    # Save the blended submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../submissions/super_blend_{timestamp}.csv"
    
    super_blend.to_csv(filename, index=False)
    logger.info(f"Saved super blend to {filename}")
    
    print(f"\nSuper blend created with weights:")
    print(f"  Advanced CatBoost: {advanced_catboost_weight:.2f}")
    print(f"  Previous Best Blend: {best_blend_weight:.2f}")
    print(f"\nSubmission file: {filename}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {filename} -m \"Super blend with advanced CatBoost\"")
    
    return filename

if __name__ == "__main__":
    create_super_blend()