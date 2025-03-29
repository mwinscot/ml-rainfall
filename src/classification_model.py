# Create a new file named classification_model.py
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def objective_xgboost_classification(trial, X, y):
    """Optuna objective for XGBoost classification."""
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
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 5.0),
        'random_state': 42,
        'use_label_encoder': False,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    model = xgb.XGBClassifier(**param)
    
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use ROC AUC score for optimization
    scores = cross_val_score(
        model, X, y, 
        scoring='roc_auc', 
        cv=kf,
        verbose=0
    )
    
    # Log the score for monitoring
    mean_score = scores.mean()
    logger.info(f"Trial AUC: {mean_score:.4f}")
    
    return mean_score

def optimize_classification_model(X, y, n_trials=20):
    """Optimize classification model hyperparameters using Optuna."""
    logger.info(f"Starting XGBoost classification optimization with {n_trials} trials")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_xgboost_classification(trial, X, y), n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best AUC: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Set objective and use_label_encoder
    best_params['objective'] = 'binary:logistic'
    best_params['use_label_encoder'] = False
    
    best_model = xgb.XGBClassifier(**best_params, random_state=42)
    
    return best_model, best_params

def evaluate_classification_model(model, X, y, X_val=None, y_val=None):
    """Evaluate classification model performance."""
    logger.info("Evaluating classification model performance")
    
    # Cross-validation evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_auc = cross_val_score(
        model, X, y, 
        scoring='roc_auc', 
        cv=kf,
        verbose=0
    )
    
    cv_scores_f1 = cross_val_score(
        model, X, y, 
        scoring='f1', 
        cv=kf,
        verbose=0
    )
    
    logger.info(f"Cross-validation AUC: {cv_scores_auc.mean():.4f} ± {cv_scores_auc.std():.4f}")
    logger.info(f"Cross-validation F1: {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}")
    
    # If validation data is provided, evaluate on it
    if X_val is not None and y_val is not None:
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        loss = log_loss(y_val, y_pred_proba)
        
        logger.info(f"Validation AUC: {auc:.4f}")
        logger.info(f"Validation Accuracy: {acc:.4f}")
        logger.info(f"Validation F1: {f1:.4f}")
        logger.info(f"Validation Log Loss: {loss:.4f}")
    
    return cv_scores_auc, cv_scores_f1