# Create ensemble_model.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold, cross_val_score

# Import modules
from src.data_processing import load_data, inspect_data, clean_data, split_data, scale_features
from src.feature_engineering import create_features
from src.model_training import optimize_model, evaluate_model
from classification_model import optimize_classification_model, evaluate_classification_model
from src.submission import create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ensemble.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_ensemble():
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
    
    # Step 6: Create binary target for classifier
    y_train_binary = (y_train > 0.5).astype(int)
    y_val_binary = (y_val > 0.5).astype(int)
    
    # Step 7: Train individual models
    logger.info("Training XGBoost Regressor")
    xgb_params = {
        'n_estimators': 749, 
        'max_depth': 8, 
        'learning_rate': 0.0656, 
        'subsample': 0.683, 
        'colsample_bytree': 0.589, 
        'min_child_weight': 7, 
        'gamma': 0.846, 
        'reg_alpha': 0.043, 
        'reg_lambda': 5.303,
        'random_state': 42
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train)
    
    logger.info("Training LightGBM Regressor")
    lgb_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': 42
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_scaled, y_train)
    
    logger.info("Training XGBoost Classifier")
    xgb_class_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.0,
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'random_state': 42
    }
    xgb_class_model = xgb.XGBClassifier(**xgb_class_params)
    xgb_class_model.fit(X_train_scaled, y_train_binary)
    
    # Step 8: Make predictions on validation data
    xgb_val_pred = xgb_model.predict(X_val_scaled)
    lgb_val_pred = lgb_model.predict(X_val_scaled)
    xgb_class_val_pred = xgb_class_model.predict_proba(X_val_scaled)[:, 1]
    
    # Step 9: Find optimal weights for blending
    weights = np.linspace(0, 1, 21)
    best_rmse = float('inf')
    best_weights = (0.33, 0.33, 0.33)  # Default equal weights
    
    for w1 in weights:
        for w2 in weights:
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
                
            # Create weighted average
            ensemble_val_pred = w1 * xgb_val_pred + w2 * lgb_val_pred + w3 * xgb_class_val_pred
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((ensemble_val_pred - y_val) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = (w1, w2, w3)
    
    logger.info(f"Best ensemble weights: {best_weights}")
    logger.info(f"Best validation RMSE: {best_rmse:.4f}")
    
    # Step 10: Train models on all data
    all_X = pd.concat([X_train, X_val])
    all_y = pd.concat([y_train, y_val])
    all_y_binary = (all_y > 0.5).astype(int)
    all_X_scaled = scaler.transform(all_X)
    
    xgb_model.fit(all_X_scaled, all_y)
    lgb_model.fit(all_X_scaled, all_y)
    xgb_class_model.fit(all_X_scaled, all_y_binary)
    
    # Step 11: Make predictions on test data
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    lgb_test_pred = lgb_model.predict(X_test_scaled)
    xgb_class_test_pred = xgb_class_model.predict_proba(X_test_scaled)[:, 1]
    
    # Step 12: Create weighted ensemble prediction
    w1, w2, w3 = best_weights
    ensemble_test_pred = w1 * xgb_test_pred + w2 * lgb_test_pred + w3 * xgb_class_test_pred
    
    # Step 13: Create submission files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create individual model submissions
    xgb_file = f"submissions/xgb_reg_{timestamp}.csv"
    lgb_file = f"submissions/lgb_reg_{timestamp}.csv"
    xgb_class_file = f"submissions/xgb_class_{timestamp}.csv"
    ensemble_file = f"submissions/ensemble_{timestamp}.csv"
    
    create_submission(test_df_cleaned['id'], xgb_test_pred, xgb_file)
    create_submission(test_df_cleaned['id'], lgb_test_pred, lgb_file)
    create_submission(test_df_cleaned['id'], xgb_class_test_pred, xgb_class_file)
    create_submission(test_df_cleaned['id'], ensemble_test_pred, ensemble_file)
    
    logger.info(f"Created 4 submission files:")
    logger.info(f" - XGBoost regression: {xgb_file}")
    logger.info(f" - LightGBM regression: {lgb_file}")
    logger.info(f" - XGBoost classification: {xgb_class_file}")
    logger.info(f" - Ensemble: {ensemble_file}")
    
    print(f"\nEnsemble submission file created: {ensemble_file}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {ensemble_file} -m \"Ensemble of XGBoost, LightGBM, and Classification models\"")

if __name__ == "__main__":
    train_ensemble()