import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Import your modules
from src.data_processing import load_data, inspect_data, clean_data, split_data, scale_features
from src.feature_engineering import create_features
from src.model_training import optimize_model, train_model, evaluate_model, predict
from src.submission import create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Step 1: Load the data
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    
    # Step 2: Inspect the data
    train_stats = inspect_data(train_df, "train")
    test_stats = inspect_data(test_df, "test")
    
    # Step 3: Feature engineering before cleaning
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Step 4: Clean the data
    train_df_cleaned = clean_data(train_df, is_train=True)
    test_df_cleaned = clean_data(test_df, is_train=False)
    
    # Step 5: Split the data
    X_train, X_val, y_train, y_val = split_data(train_df_cleaned)
    
    # Step 6: Scale the features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, 
        X_val, 
        test_df_cleaned.drop(columns=['id'])
    )
    
    # Step 7: Check for potential issues before optimization
    print("Train target stats:", y_train.describe())
    print("Validation target stats:", y_val.describe())
    
    # Step 8: Optimize the model (with fewer trials for quicker iteration)
    best_model, best_params = optimize_model(X_train_scaled, y_train, n_trials=20)
    print("Best parameters:", best_params)
    
    # Step 9: Evaluate the model
    cv_scores = evaluate_model(best_model, X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Step 10: Train the final model on all training data
    all_X = pd.concat([X_train, X_val])
    all_y = pd.concat([y_train, y_val])
    all_X_scaled = scaler.transform(all_X)
    final_model = train_model(best_model, all_X_scaled, all_y)
    
    # Step 11: Make predictions on the test set
    test_predictions = predict(final_model, X_test_scaled)
    
    # Step 12: Create a submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"submissions/submission_{timestamp}.csv"
    submission = create_submission(test_df_cleaned['id'], test_predictions, submission_file)
    
    logger.info(f"Submission file created: {submission_file}")
    
    # Step 13: Print submission instructions
    print(f"\nSubmission file created: {submission_file}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {submission_file} -m \"XGBoost model\"")

if __name__ == "__main__":
    main()