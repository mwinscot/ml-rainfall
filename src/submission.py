import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("submission.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_submission(test_ids, predictions, filename=None):
    """Create a submission file for Kaggle."""
    logger.info("Creating submission file")
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_ids,
        'target': predictions
    })
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submissions/submission_{timestamp}.csv"
    
    # Save submission
    submission_df.to_csv(filename, index=False)
    logger.info(f"Submission saved to {filename}")
    
    return submission_df

def submit_to_kaggle(filename, message="Automated submission"):
    """Submit to Kaggle using the Kaggle API."""
    import subprocess
    import os
    
    logger.info(f"Submitting {filename} to Kaggle")
    
    try:
        # Check if file exists
        if not os.path.exists(filename):
            logger.error(f"Submission file {filename} not found")
            return False
        
        # Run Kaggle CLI command
        cmd = f"kaggle competitions submit -c playground-series-s5e3 -f {filename} -m \"{message}\""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Submission successful")
            logger.info(result.stdout)
            return True
        else:
            logger.error("Submission failed")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error submitting to Kaggle: {e}")
        return False