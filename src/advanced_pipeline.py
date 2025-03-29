import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import logging
from datetime import datetime

# Import our modules
from src.data_processing import load_data, clean_data, split_data, scale_features
from src.feature_engineering import create_features
from advanced_features import add_advanced_features
from src.submission import create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cross_validation(X, y, model, n_folds=5, classification=False):
    """Perform cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    oof_predictions = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        
        if classification:
            y_pred = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_pred)
        else:
            y_pred = model.predict(X_val_fold)
            score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        
        scores.append(score)
        oof_predictions[val_idx] = y_pred
        
        logger.info(f"Fold {fold+1}/{n_folds} - {'AUC' if classification else 'RMSE'}: {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"CV {'AUC' if classification else 'RMSE'}: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score, oof_predictions

def feature_selection(X_train, y_train, X_val, X_test, threshold=0.01):
    """Select important features."""
    logger.info("Performing feature selection")
    
    # Train a model for feature importance
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Select features
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    
    # Transform data
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    logger.info(f"Top 10 features: {selected_features[:10].tolist()}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

def train_stacked_models(X_train, y_train, X_val, y_val, classification=False):
    """Train multiple models for stacking."""
    logger.info(f"Training {'classification' if classification else 'regression'} models for stacking")
    
    models = {}
    oof_predictions = {}
    
    # Define models
    if classification:
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            random_state=42
        )
        
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        )
        
        models['cat'] = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            subsample=0.7,
            random_seed=42,
            verbose=0
        )
    else:
        models['xgb'] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            random_state=42
        )
        
        models['lgb'] = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        )
        
        models['cat'] = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            subsample=0.7,
            random_seed=42,
            verbose=0
        )
    
    # Train each model
    for name, model in models.items():
        logger.info(f"Training {name} model")
        
        if classification:
            model.fit(X_train, y_train)
            train_preds = model.predict_proba(X_train)[:, 1]
            val_preds = model.predict_proba(X_val)[:, 1]
            
            train_score = roc_auc_score(y_train, train_preds)
            val_score = roc_auc_score(y_val, val_preds)
            
            logger.info(f"{name} - Train AUC: {train_score:.4f}, Val AUC: {val_score:.4f}")
        else:
            model.fit(X_train, y_train)
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            
            train_score = np.sqrt(mean_squared_error(y_train, train_preds))
            val_score = np.sqrt(mean_squared_error(y_val, val_preds))
            
            logger.info(f"{name} - Train RMSE: {train_score:.4f}, Val RMSE: {val_score:.4f}")
        
        oof_predictions[name] = val_preds
    
    return models, oof_predictions

def train_meta_model(X_meta, y_meta, classification=False):
    """Train a meta-model for stacking."""
    logger.info(f"Training {'classification' if classification else 'regression'} meta-model")
    
    if classification:
        meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=1.0,
            objective='binary:logistic',
            random_state=42
        )
    else:
        meta_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42
        )
    
    meta_model.fit(X_meta, y_meta)
    return meta_model

def predict_with_stacked_models(models, meta_model, X_test, classification=False):
    """Generate predictions using stacked models."""
    logger.info("Generating stacked predictions")
    
    # Generate base model predictions
    base_predictions = np.zeros((X_test.shape[0], len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        if classification:
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test)
        
        base_predictions[:, i] = preds
    
    # Generate meta-model predictions
    final_predictions = meta_model.predict(base_predictions)
    
    return final_predictions

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
    
    # Check if it's a classification problem
    is_classification = (y_train.nunique() <= 2)
    
    if is_classification:
        logger.info("Treating this as a classification problem")
        y_train_binary = (y_train > 0.5).astype(int)
        y_val_binary = (y_val > 0.5).astype(int)
        class_distribution = y_train_binary.value_counts(normalize=True)
        logger.info(f"Class distribution: {class_distribution.to_dict()}")
    else:
        logger.info("Treating this as a regression problem")
        y_train_binary = y_train
        y_val_binary = y_val
    
    # Step 6: Add advanced features
    X_train_advanced, X_val_advanced, X_test_advanced = add_advanced_features(
        X_train_scaled, X_val_scaled, X_test_scaled
    )
    
    # Step 7: Feature selection
    X_train_selected, X_val_selected, X_test_selected, selected_features = feature_selection(
        X_train_advanced, y_train_binary, X_val_advanced, X_test_advanced
    )
    
    # Step 8: Train base models for stacking
    base_models, oof_predictions = train_stacked_models(
        X_train_selected, y_train_binary, X_val_selected, y_val_binary, 
        classification=is_classification
    )
    
    # Step 9: Prepare meta-features
    X_meta = np.column_stack([oof_predictions[name] for name in base_models.keys()])
    
    # Step 10: Train meta-model
    meta_model = train_meta_model(X_meta, y_val_binary, classification=is_classification)
    
    # Step 11: Train final models on all data
    logger.info("Training final models on all data")
    all_X = pd.concat([X_train, X_val])
    all_y = pd.concat([y_train, y_val])
    all_X_scaled = scaler.transform(all_X)
    
    # Convert to binary for classification
    if is_classification:
        all_y_binary = (all_y > 0.5).astype(int)
    else:
        all_y_binary = all_y
    
    # Add advanced features
    all_X_advanced = add_advanced_features(all_X_scaled)
    
    # Feature selection
    selector = SelectFromModel(xgb.XGBRegressor(random_state=42), threshold=0.01)
    selector.fit(all_X_advanced, all_y_binary)
    all_X_selected = selector.transform(all_X_advanced)
    
    # Train final base models
    final_base_models = {}
    for name, model_template in base_models.items():
        logger.info(f"Training final {name} model")
        if hasattr(model_template, 'get_params'):
            final_model = type(model_template)(**model_template.get_params())
        else:
            final_model = type(model_template)()
        
        final_model.fit(all_X_selected, all_y_binary)
        final_base_models[name] = final_model
    
    # Step 12: Generate predictions
    final_predictions = predict_with_stacked_models(
        final_base_models, meta_model, X_test_selected, classification=is_classification
    )
    
    # Step 13: Create submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"submissions/advanced_stack_{timestamp}.csv"
    submission = create_submission(test_df_cleaned['id'], final_predictions, submission_file)
    
    logger.info(f"Submission file created: {submission_file}")
    
    # Step 14: Also create individual model submissions
    for name, model in final_base_models.items():
        if is_classification:
            predictions = model.predict_proba(X_test_selected)[:, 1]
        else:
            predictions = model.predict(X_test_selected)
        
        model_submission_file = f"submissions/{name}_{timestamp}.csv"
        create_submission(test_df_cleaned['id'], predictions, model_submission_file)
        logger.info(f"Created individual model submission: {model_submission_file}")
    
if __name__ == "__main__":
    main()