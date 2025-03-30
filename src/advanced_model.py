import pandas as pd
import numpy as np
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Import your modules
from data_processing import load_data, clean_data, split_data, scale_features
from feature_engineering import create_features
from advanced_features import enhanced_feature_engineering, select_important_features
from submission import create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_advanced_model():
    """Train an advanced model with enhanced features."""
    # Step 1: Load the data
    logger.info("Loading data")
    train_df, test_df = load_data('../data/train.csv', '../data/test.csv')
    
    # Step 2: Basic feature engineering
    logger.info("Applying basic feature engineering")
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Step 3: Enhanced feature engineering
    logger.info("Applying enhanced feature engineering")
    train_df, test_df = enhanced_feature_engineering(train_df, test_df)
    
    # Step 4: Clean the data
    logger.info("Cleaning data")
    train_df_cleaned = clean_data(train_df, is_train=True)
    test_df_cleaned = clean_data(test_df, is_train=False)
    
    # Step 5: Split the data
    logger.info("Splitting data")
    X_train, X_val, y_train, y_val = split_data(train_df_cleaned)
    
    # Step 6: Scale the features
    logger.info("Scaling features")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, 
        X_val, 
        test_df_cleaned.drop(columns=['id'])
    )
    
    # Step 7: Convert target to binary (for classification)
    logger.info("Converting target to binary")
    y_train_binary = (y_train > 0.5).astype(int)
    y_val_binary = (y_val > 0.5).astype(int)
    
    # Step 8: Feature selection
    logger.info("Selecting important features")
    selected_data = select_important_features(
        X_train_scaled, y_train_binary, X_val_scaled, X_test_scaled, threshold=0.005
    )
    
    X_train_selected, X_val_selected, X_test_selected, selected_features = selected_data
    
    # Log shape after selection
    logger.info(f"Shape after feature selection - Train: {X_train_selected.shape}, Val: {X_val_selected.shape}")
    
    # Step 9: Train best model with optimized parameters
    logger.info("Training model with best parameters")
    
    # Use your best parameters from previous hyperparameter tuning
    best_params = {
        'n_estimators': 412, 
        'max_depth': 4, 
        'learning_rate': 0.012875008626555221, 
        'subsample': 0.9078242243681212, 
        'colsample_bytree': 0.591862055398215, 
        'min_child_weight': 8, 
        'gamma': 0.446296513574009, 
        'reg_alpha': 5.302523442648221e-06, 
        'reg_lambda': 8.439073332306782, 
        'scale_pos_weight': 7.675184588140441,
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**best_params)
    
    # Train on cross-validation to get a more robust model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in kf.split(X_train_selected):
        # Split data
        cv_X_train = X_train_selected.iloc[train_idx]
        cv_y_train = y_train_binary.iloc[train_idx]
        cv_X_test = X_train_selected.iloc[test_idx]
        cv_y_test = y_train_binary.iloc[test_idx]
        
        # Train model
        model.fit(cv_X_train, cv_y_train)
        
        # Evaluate
        cv_preds = model.predict_proba(cv_X_test)[:, 1]
        cv_score = roc_auc_score(cv_y_test, cv_preds)
        cv_scores.append(cv_score)
    
    logger.info(f"Cross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final model on all training data
    logger.info("Training final model on all data")
    model.fit(X_train_selected, y_train_binary)
    
    # Evaluate on validation data
    val_preds = model.predict_proba(X_val_selected)[:, 1]
    val_score = roc_auc_score(y_val_binary, val_preds)
    logger.info(f"Validation AUC: {val_score:.4f}")
    
    # Step 10: Train on all data
    logger.info("Training on all data for final prediction")

    # Get only the numeric columns the scaler was trained on
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    logger.info(f"Number of numeric features the scaler was trained on: {len(numeric_cols)}")

    # Concatenate training and validation data
    all_X = pd.concat([X_train, X_val])
    all_y = pd.concat([y_train, y_val])
    all_y_binary = (all_y > 0.5).astype(int)
    
    # Create a separate DataFrame for scaling with only numeric columns
    all_X_numeric = all_X[numeric_cols].copy()
    logger.info(f"Scaling only numeric columns, shape: {all_X_numeric.shape}")
    
    # Apply scaling only to numeric columns
    all_X_numeric_scaled = pd.DataFrame(
        scaler.transform(all_X_numeric),
        columns=numeric_cols,
        index=all_X.index
    )
    
    # Get categorical columns if any
    categorical_cols = [col for col in all_X.columns if col not in numeric_cols]
    if categorical_cols:
        logger.info(f"Found {len(categorical_cols)} categorical columns")
        # Create all_X_scaled by combining scaled numeric and original categorical
        all_X_scaled = pd.concat([all_X_numeric_scaled, all_X[categorical_cols]], axis=1)
    else:
        all_X_scaled = all_X_numeric_scaled
    
    logger.info(f"Combined scaled data shape: {all_X_scaled.shape}")

    # Use the same columns as X_train_selected
    if hasattr(X_train_selected, 'columns'):
        selected_cols = X_train_selected.columns.tolist()
        logger.info(f"Using {len(selected_cols)} pre-selected features from training phase")
        
        # Find common columns between all_X_scaled and selected_cols
        common_cols = [col for col in selected_cols if col in all_X_scaled.columns]
        logger.info(f"Found {len(common_cols)} common columns for feature selection")
        
        all_X_selected = all_X_scaled[common_cols]
    else:
        # If X_train_selected is not a DataFrame, use the previously fitted selector
        logger.info("Using feature selector from training phase")
        from sklearn.feature_selection import SelectFromModel
        selector = SelectFromModel(model, threshold=0.005, prefit=True)
        all_X_selected = selector.transform(all_X_scaled)
    
    # Train final model
    logger.info(f"Training final model with shape: {all_X_selected.shape}")
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(all_X_selected, all_y_binary)
    
    # Handle test data in the same way
    logger.info("Preparing test data for prediction")
    test_features = test_df_cleaned.drop(columns=['id'])
    
    # Apply scaling only to numeric columns in test data
    test_numeric_cols = [col for col in numeric_cols if col in test_features.columns]
    logger.info(f"Scaling {len(test_numeric_cols)} numeric test columns")
    
    test_numeric_scaled = pd.DataFrame(
        scaler.transform(test_features[test_numeric_cols]),
        columns=test_numeric_cols,
        index=test_features.index
    )
    
    # Get categorical columns in test data
    test_categorical_cols = [col for col in test_features.columns if col not in numeric_cols]
    if test_categorical_cols:
        # Create X_test_scaled by combining scaled numeric and original categorical
        X_test_scaled_df = pd.concat([test_numeric_scaled, test_features[test_categorical_cols]], axis=1)
    else:
        X_test_scaled_df = test_numeric_scaled
    
    # Apply feature selection to test data
    if hasattr(X_train_selected, 'columns'):
        # Use the same selected columns for test data
        common_test_cols = [col for col in selected_cols if col in X_test_scaled_df.columns]
        logger.info(f"Using {len(common_test_cols)} common selected columns for test data")
        X_test_selected_final = X_test_scaled_df[common_test_cols]
    else:
        # Use the same selector
        X_test_selected_final = selector.transform(X_test_scaled_df)
    
    # Step 11: Make predictions on test data
    logger.info(f"Making predictions with final model, test data shape: {X_test_selected_final.shape}")
    test_preds = final_model.predict_proba(X_test_selected_final)[:, 1]
    
    # Step 12: Create submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"../submissions/advanced_model_{timestamp}.csv"
    
    create_submission(test_df_cleaned['id'], test_preds, submission_file)
    
    logger.info(f"Submission file created: {submission_file}")
    
    print(f"\nAdvanced model submission file created: {submission_file}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {submission_file} -m \"Advanced model with enhanced features\"")
    
    return submission_file

if __name__ == "__main__":
    train_advanced_model()