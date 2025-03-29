# Create an updated model_training.py file
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def objective_xgboost(trial, X, y):
    """Optuna objective for XGBoost."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'eval_metric': 'rmse'
    }
    
    model = xgb.XGBRegressor(**param)
    
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use negative RMSE as the score (higher is better for Optuna maximization)
    scores = -cross_val_score(
        model, X, y, 
        scoring='neg_root_mean_squared_error', 
        cv=kf,
        verbose=0
    )
    
    # Log the score for monitoring
    mean_score = scores.mean()
    logger.info(f"Trial RMSE: {mean_score:.4f}")
    
    return -mean_score  # Return negative RMSE for maximization

def optimize_model(X, y, model_type='xgboost', n_trials=100):
    """Optimize model hyperparameters using Optuna."""
    logger.info(f"Starting {model_type} optimization with {n_trials} trials")
    
    if model_type == 'xgboost':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=n_trials)
        
        best_params = study.best_params
        best_score = -study.best_value  # Convert back to RMSE
        
        logger.info(f"Best RMSE: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
    
    return best_model, best_params

def evaluate_model(model, X, y, X_val=None, y_val=None):
    """Evaluate the model performance."""
    logger.info("Evaluating model performance")
    
    # Cross-validation evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = -cross_val_score(
        model, X, y, 
        scoring='neg_root_mean_squared_error', 
        cv=kf,
        verbose=0
    )
    
    logger.info(f"Cross-validation RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Validation scores: {cv_scores}")
    
    # If validation data is provided, evaluate on it
    if X_val is not None and y_val is not None:
        model.fit(X, y)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        logger.info(f"Validation RMSE: {rmse:.4f}")
        logger.info(f"Validation MAE: {mae:.4f}")
    
    return cv_scores

def train_model(model, X, y):
    """Train the model on the entire dataset."""
    logger.info("Training final model")
    model.fit(X, y)
    return model

def predict(model, X):
    """Make predictions with the trained model."""
    return model.predict(X)