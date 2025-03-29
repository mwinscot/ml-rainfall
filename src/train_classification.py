import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import modules
from src.data_processing import load_data, inspect_data, clean_data, split_data, scale_features
from src.feature_engineering import create_features
from classification_model import optimize_classification_model, evaluate_classification_model
from src.submission import create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Step 1: Load the data
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    
    # Step 2: Feature engineering
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Step 3: Clean the data
    train_df_cleaned = clean_data(train_df, is_train=True)
    test_df_cleaned = clean_data(test_df, is_train=False)
    
    # Step 4: Split the data
    X_train, X_val, y_train, y_val = split_data(train_df_cleaned)
    
    # Step 5: Scale the features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, 
        X_val, 
        test_df_cleaned.drop(columns=['id'])
    )
    
    # Step 6: Convert target to binary if it's not already
    print("Train target stats:", y_train.describe())
    print("Validation target stats:", y_val.describe())
    
    # Ensure binary classification
    y_train_binary = (y_train > 0.5).astype(int)
    y_val_binary = (y_val > 0.5).astype(int)
    
    print("Binary train class distribution:")
    print(y_train_binary.value_counts(normalize=True))
    
    # Step 7: Optimize classification model
    best_model, best_params = optimize_classification_model(X_train_scaled, y_train_binary, n_trials=20)
    print("Best parameters:", best_params)
    
    # Step 8: Evaluate the model
    cv_scores_auc, cv_scores_f1 = evaluate_classification_model(
        best_model, 
        X_train_scaled, 
        y_train_binary, 
        X_val_scaled, 
        y_val_binary
    )
    
    # Step 9: Train the final model on all training data
    logger.info("Training final model")
    all_X = pd.concat([X_train, X_val])
    all_y = pd.concat([y_train, y_val])
    all_X_scaled = scaler.transform(all_X)
    all_y_binary = (all_y > 0.5).astype(int)
    
    best_model.fit(all_X_scaled, all_y_binary)
    
    # Step 10: Make predictions on the test set
    test_predictions_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    logger.info("Predictions created")
    
    # Step 11: Create a submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"submissions/class_submission_{timestamp}.csv"
    submission = create_submission(test_df_cleaned['id'], test_predictions_proba, submission_file)
    
    logger.info(f"Submission file created: {submission_file}")
    
    # Step 12: Print submission instructions
    print(f"\nClassification submission file created: {submission_file}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {submission_file} -m \"XGBoost classification model\"")

if __name__ == "__main__":
    main()