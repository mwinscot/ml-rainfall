import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_weighted_blend():
    """Create a weighted blend of multiple model predictions."""
    
    # Step 1: Load the prediction files
    logger.info("Loading prediction files")
    
    # Get submissions directory
    submissions_dir = "submissions"
    
    # Prompt user to select files for blending
    print("\nAvailable submission files:")
    submission_files = [f for f in os.listdir(submissions_dir) if f.endswith('.csv')]
    submission_files.sort(key=lambda x: os.path.getmtime(os.path.join(submissions_dir, x)), reverse=True)
    
    for i, file in enumerate(submission_files[:10]):  # Show only 10 most recent
        print(f"{i+1}. {file}")
    
    # Get user input for files to blend
    selected_indices = input("\nEnter the numbers of files to blend (comma-separated): ")
    selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(',')]
    
    selected_files = [submission_files[idx] for idx in selected_indices if 0 <= idx < len(submission_files)]
    
    if len(selected_files) < 2:
        print("Need at least 2 files for blending. Exiting.")
        return
    
    # Load the prediction files
    predictions = {}
    for file in selected_files:
        file_path = os.path.join(submissions_dir, file)
        predictions[file] = pd.read_csv(file_path)
        print(f"Loaded {file}")
    
    # Step 2: Determine weights
    print("\nEnter weights for each model (should sum to 1.0):")
    weights = {}
    
    for i, file in enumerate(selected_files):
        weight = float(input(f"Weight for {file} (default: {1.0/len(selected_files):.2f}): ") or 1.0/len(selected_files))
        weights[file] = weight
    
    # Normalize weights to ensure they sum to 1.0
    total_weight = sum(weights.values())
    for file in weights:
        weights[file] /= total_weight
    
    print("\nNormalized weights:")
    for file, weight in weights.items():
        print(f"{file}: {weight:.4f}")
    
    # Step 3: Calculate weighted blend
    logger.info("Creating weighted blend")
    
    # Start with first file as template
    blend_df = predictions[selected_files[0]].copy()
    blend_df['target'] = 0
    
    # Add weighted predictions
    for file, weight in weights.items():
        blend_df['target'] += weight * predictions[file]['target']
    
    # Step 4: Save the blended prediction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create weight string for filename
    weight_str = "_".join([f"{w:.2f}".replace('.', '') for w in weights.values()])
    filename = f'submissions/blend_{weight_str}_{timestamp}.csv'
    
    blend_df.to_csv(filename, index=False)
    logger.info(f"Saved blended submission to {filename}")
    print(f"\nSaved blended submission to {filename}")
    
    # Step 5: Print Kaggle submission command
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {filename} -m \"Blend of {len(selected_files)} models\"")
    
    return filename

if __name__ == "__main__":
    print("Model Prediction Blender")
    print("=======================")
    create_weighted_blend()