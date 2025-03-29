import os
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the result."""
    logger.info(f"Running: {description}")
    print(f"\n=== {description} ===")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        logger.info(f"Completed: {description}")
        print(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        print(f"Error: {e}")
        return False

def main():
    print("\nKaggle Competition Workflow")
    print("==========================")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Menu options
    while True:
        print("\nOptions:")
        print("1. Download competition data")
        print("2. Run basic model")
        print("3. Run classification model")
        print("4. Run ensemble model")
        print("5. Run advanced pipeline")
        print("6. Run feature analysis")
        print("7. Tune XGBoost hyperparameters")
        print("8. Tune LightGBM hyperparameters")
        print("9. Tune CatBoost hyperparameters")
        print("10. Create submission blend")
        print("11. Submit to Kaggle")
        print("12. Check Kaggle leaderboard")
        print("13. Exit")
        
        choice = input("\nEnter your choice (1-13): ")
        
        if choice == '1':
            run_command("kaggle competitions download -c playground-series-s5e3 -p data/", 
                       "Downloading competition data")
            run_command("powershell -Command \"Expand-Archive -Path data/playground-series-s5e3.zip -DestinationPath data/ -Force\"",
                       "Extracting data")
        
        elif choice == '2':
            run_command("python main.py", 
                       "Running basic model")
        
        elif choice == '3':
            run_command("python train_classification.py", 
                       "Running classification model")
        
        elif choice == '4':
            run_command("python ensemble_model.py", 
                       "Running ensemble model")
        
        elif choice == '5':
            run_command("python advanced_pipeline.py", 
                       "Running advanced pipeline")
        
        elif choice == '6':
            run_command("python feature_analysis.py", 
                       "Running feature analysis")
        
        elif choice == '7':
            trials = input("Enter number of trials (default: 50): ") or "50"
            mode = input("Classification or regression? (c/r, default: r): ") or "r"
            
            if mode.lower() == 'c':
                run_command(f"python hyperparameter_tuning.py --model xgboost --trials {trials} --classification",
                           "Tuning XGBoost (classification)")
            else:
                run_command(f"python hyperparameter_tuning.py --model xgboost --trials {trials}",
                           "Tuning XGBoost (regression)")
        
        elif choice == '8':
            trials = input("Enter number of trials (default: 50): ") or "50"
            mode = input("Classification or regression? (c/r, default: r): ") or "r"
            
            if mode.lower() == 'c':
                run_command(f"python hyperparameter_tuning.py --model lightgbm --trials {trials} --classification",
                           "Tuning LightGBM (classification)")
            else:
                run_command(f"python hyperparameter_tuning.py --model lightgbm --trials {trials}",
                           "Tuning LightGBM (regression)")
        
        elif choice == '9':
            trials = input("Enter number of trials (default: 50): ") or "50"
            mode = input("Classification or regression? (c/r, default: r): ") or "r"
            
            if mode.lower() == 'c':
                run_command(f"python hyperparameter_tuning.py --model catboost --trials {trials} --classification",
                           "Tuning CatBoost (classification)")
            else:
                run_command(f"python hyperparameter_tuning.py --model catboost --trials {trials}",
                           "Tuning CatBoost (regression)")
        
        elif choice == '10':
            run_command("python make_submissions.py", 
                       "Creating submission blend")
        
        elif choice == '11':
            # List submission files
            submission_dir = "submissions"
            files = os.listdir(submission_dir)
            submission_files = [f for f in files if f.endswith('.csv')]
            
            # Sort by creation time (newest first)
            submission_files.sort(key=lambda x: os.path.getctime(os.path.join(submission_dir, x)), reverse=True)
            
            print(f"\nFound {len(submission_files)} submission files:")
            for i, file in enumerate(submission_files[:10]):  # Show only 10 most recent
                file_path = os.path.join(submission_dir, file)
                creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                print(f"{i+1}. {file} - Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if submission_files:
                file_idx = int(input("\nEnter the number of the file to submit (or 0 to cancel): ")) - 1
                if 0 <= file_idx < len(submission_files):
                    message = input("Enter a submission message: ")
                    file_path = os.path.join(submission_dir, submission_files[file_idx])
                    run_command(f"kaggle competitions submit -c playground-series-s5e3 -f \"{file_path}\" -m \"{message}\"",
                               "Submitting to Kaggle")
            else:
                print("No submission files found")
        
        elif choice == '12':
            run_command("kaggle competitions leaderboard playground-series-s5e3 --show",
                       "Checking Kaggle leaderboard")
        
        elif choice == '13':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()