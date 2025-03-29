import os
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("submissions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def submit_to_kaggle(filename, message):
    """Submit a file to Kaggle."""
    cmd = f"kaggle competitions submit -c playground-series-s5e3 -f {filename} -m \"{message}\""
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully submitted {filename}")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Failed to submit {filename}")
            logger.error(result.stderr)
            return False
    
    except Exception as e:
        logger.error(f"Error during submission: {e}")
        return False

def list_submissions():
    """List all submissions in the submissions directory."""
    submission_dir = "submissions"
    files = os.listdir(submission_dir)
    submission_files = [f for f in files if f.endswith('.csv')]
    
    # Sort by creation time (newest first)
    submission_files.sort(key=lambda x: os.path.getctime(os.path.join(submission_dir, x)), reverse=True)
    
    print(f"\nFound {len(submission_files)} submission files:")
    for i, file in enumerate(submission_files):
        file_path = os.path.join(submission_dir, file)
        creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        print(f"{i+1}. {file} - Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')} - Size: {file_size:.2f} KB")
    
    return submission_files

def create_blend_submission(files, weights=None):
    """Create a blended submission from multiple files."""
    if weights is None:
        weights = [1/len(files)] * len(files)
    
    if len(files) != len(weights):
        raise ValueError("Number of files and weights must match")
    
    submissions = []
    for file in files:
        file_path = f"submissions/{file}"
        df = pd.read_csv(file_path)
        submissions.append(df)
    
    # Check that all submissions have the same IDs in the same order
    id_check = all(submissions[0]['id'].equals(df['id']) for df in submissions[1:])
    if not id_check:
        raise ValueError("IDs in submission files do not match")
    
    # Create blended prediction
    blend_df = submissions[0].copy()
    blend_df['target'] = 0
    
    for i, df in enumerate(submissions):
        blend_df['target'] += df['target'] * weights[i]
    
    # Save blended submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blend_file = f"submissions/blend_{timestamp}.csv"
    blend_df.to_csv(blend_file, index=False)
    
    logger.info(f"Created blended submission: {blend_file}")
    print(f"Created blended submission: {blend_file}")
    
    # Create message with weights
    weight_str = ', '.join([f"{w:.2f}" for w in weights])
    message = f"Blend of {len(files)} submissions with weights: {weight_str}"
    
    return blend_file, message

def main():
    print("\nKaggle Submission Manager")
    print("========================")
    
    while True:
        print("\nOptions:")
        print("1. List available submission files")
        print("2. Submit a single file to Kaggle")
        print("3. Create a blended submission")
        print("4. Check Kaggle leaderboard")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            submission_files = list_submissions()
        
        elif choice == '2':
            submission_files = list_submissions()
            if not submission_files:
                print("No submission files found")
                continue
                
            file_idx = int(input("\nEnter the number of the file to submit: ")) - 1
            if 0 <= file_idx < len(submission_files):
                message = input("Enter a submission message: ")
                file_path = f"submissions/{submission_files[file_idx]}"
                submit_to_kaggle(file_path, message)
            else:
                print("Invalid file number")
        
        elif choice == '3':
            submission_files = list_submissions()
            if not submission_files:
                print("No submission files found")
                continue
                
            selected_files = input("\nEnter file numbers to blend (comma-separated): ")
            try:
                file_indices = [int(idx.strip()) - 1 for idx in selected_files.split(',')]
                selected_submissions = [submission_files[idx] for idx in file_indices if 0 <= idx < len(submission_files)]
                
                if not selected_submissions:
                    print("No valid files selected")
                    continue
                
                weights_input = input(f"Enter weights for {len(selected_submissions)} files (comma-separated, or blank for equal weights): ")
                if weights_input.strip():
                    weights = [float(w.strip()) for w in weights_input.split(',')]
                    # Normalize weights to sum to 1
                    weights = [w / sum(weights) for w in weights]
                else:
                    weights = None
                
                blend_file, message = create_blend_submission(selected_submissions, weights)
                
                submit_now = input("Submit this blend to Kaggle now? (y/n): ")
                if submit_now.lower() == 'y':
                    submit_to_kaggle(blend_file, message)
            
            except Exception as e:
                print(f"Error creating blend: {e}")
        
        elif choice == '4':
            try:
                subprocess.run("kaggle competitions leaderboard playground-series-s5e3 --show", shell=True)
            except Exception as e:
                print(f"Error checking leaderboard: {e}")
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()