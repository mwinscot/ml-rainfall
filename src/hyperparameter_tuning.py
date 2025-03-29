import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score
import logging
from datetime import datetime

# Import our modules
from src.data_processing import load_data, clean_data, split_data, scale_features
from src.feature_engineering import create_features
from src.submission import create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperopt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def objective_xgboost(trial, X, y, classification=False):
    """Optuna objective for XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }
    
    if classification:
        params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 10.0)
        params['objective'] = 'binary:logistic'
        model = xgb.XGBClassifier(**params)
        scoring = 'roc_auc'
    else:
        model = xgb.XGBRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=kf)
    
    if classification:
        return scores.mean()
    else:
        return -scores.mean()  # Negate RMSE for maximization

def objective_lightgbm(trial, X, y, classification=False):
    """Optuna objective for LightGBM."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }
    
    if classification:
        params['is_unbalance'] = trial.suggest_categorical('is_unbalance', [True, False])
        model = lgb.LGBMClassifier(**params)
        scoring = 'roc_auc'
    else:
        model = lgb.LGBMRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=kf)
    
    if classification:
        return scores.mean()
    else:
        return -scores.mean()  # Negate RMSE for maximization

def objective_catboost(trial, X, y, classification=False):
    """Optuna objective for CatBoost."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'random_seed': 42,
        'verbose': 0
    }
    
    if classification:
        model = CatBoostClassifier(**params)
        scoring = 'roc_auc'
    else:
        model = CatBoostRegressor(**params)
        scoring = 'neg_root_mean_squared_error'
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=kf)
    
    if classification:
        return scores.mean()
    else:
        return -scores.mean()  # Negate RMSE for maximization

def optimize_hyperparameters(X, y, model_type, n_trials, classification=False):
    """Run hyperparameter optimization for the specified model type."""
    logger.info(f"Running {model_type} hyperparameter optimization with {n_trials} trials")
    
    if model_type == 'xgboost':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_xgboost(trial, X, y, classification), n_trials=n_trials)
    elif model_type == 'lightgbm':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_lightgbm(trial, X, y, classification), n_trials=n_trials)
    elif model_type == 'catboost':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_catboost(trial, X, y, classification), n_trials=n_trials)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    best_params = study.best_params
    best_value = study.best_value
    
    if not classification:
        best_value = -best_value  # Convert back to RMSE
        logger.info(f"Best RMSE: {best_value:.4f}")
    else:
        logger.info(f"Best AUC: {best_value:.4f}")
    
    logger.info(f"Best parameters: {best_params}")
    
    return best_params, best_value

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, test_ids, model_type, params, classification=False):
    """Train a model with the given parameters and evaluate it."""
    logger.info(f"Training {model_type} model with optimized parameters")
    
    if model_type == 'xgboost':
        if classification:
            params['objective'] = 'binary:logistic'
            model = xgb.XGBClassifier(**params, random_state=42)
        else:
            model = xgb.XGBRegressor(**params, random_state=42)
    elif model_type == 'lightgbm':
        if classification:
            model = lgb.LGBMClassifier(**params, random_state=42)
        else:
            model = lgb.LGBMRegressor(**params, random_state=42)
    elif model_type == 'catboost':
        if classification:
            model = CatBoostClassifier(**params, random_seed=42, verbose=0)
        else:
            model = CatBoostRegressor(**params, random_seed=42, verbose=0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train on training data
    model.fit(X_train, y_train)
    
    # Evaluate on validation data
    if classification:
        val_pred = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_pred)
        logger.info(f"Validation AUC: {val_score:.4f}")
    else:
        val_pred = model.predict(X_val)
        val_score = np.sqrt(mean_squared_error(y_val, val_pred))
        logger.info(f"Validation RMSE: {val_score:.4f}")
    
    # Train on all data
    logger.info("Training final model on all data")
    all_X = np.vstack([X_train, X_val])
    all_y = np.concatenate([y_train, y_val])
    
    if model_type == 'xgboost':
        if classification:
            final_model = xgb.XGBClassifier(**params, random_state=42)
        else:
            final_model = xgb.XGBRegressor(**params, random_state=42)
    elif model_type == 'lightgbm':
        if classification:
            final_model = lgb.LGBMClassifier(**params, random_state=42)
        else:
            final_model = lgb.LGBMRegressor(**params, random_state=42)
    elif model_type == 'catboost':
        if classification:
            final_model = CatBoostClassifier(**params, random_seed=42, verbose=0)
        else:
            final_model = CatBoostRegressor(**params, random_seed=42, verbose=0)
    
    final_model.fit(all_X, all_y)
    
    # Make predictions on test data
    if classification:
        test_pred = final_model.predict_proba(X_test)[:, 1]
    else:
        test_pred = final_model.predict(X_test)
    
    # Create submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"submissions/{model_type}_opt_{timestamp}.csv"
    
    create_submission(test_ids, test_pred, submission_file)
    logger.info(f"Submission file created: {submission_file}")
    
    print(f"\nOptimized {model_type} submission file created: {submission_file}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {submission_file} -m \"Optimized {model_type} model\"")
    
    return final_model, val_score, submission_file

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Kaggle competition')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'lightgbm', 'catboost'], 
                        help='Model type to optimize')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--classification', action='store_true', help='Treat as classification problem')
    args = parser.parse_args()
    
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
    
    # Convert to numpy arrays for faster processing
    X_train_np = X_train_scaled
    X_val_np = X_val_scaled
    X_test_np = X_test_scaled
    
    if isinstance(X_train_scaled, pd.DataFrame):
        X_train_np = X_train_scaled.values
        X_val_np = X_val_scaled.values
        X_test_np = X_test_scaled.values
    
    # Check if it's a classification problem
    if args.classification:
        logger.info("Treating as a classification problem")
        y_train_processed = (y_train > 0.5).astype(int)
        y_val_processed = (y_val > 0.5).astype(int)
    else:
        logger.info("Treating as a regression problem")
        y_train_processed = y_train
        y_val_processed = y_val
    
    # Step 6: Run hyperparameter optimization
    best_params, best_score = optimize_hyperparameters(
        X_train_np, y_train_processed, args.model, args.trials, args.classification
    )
    
    # Step 7: Train and evaluate the model with the best parameters
    final_model, val_score, submission_file = train_and_evaluate(
        X_train_np, y_train_processed, X_val_np, y_val_processed, 
        X_test_np, test_df_cleaned['id'], args.model, best_params, args.classification
    )
    
    # Step 8: Save the best parameters
    import json
    params_file = f"best_params_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Best parameters saved to {params_file}")

if __name__ == "__main__":
    main()