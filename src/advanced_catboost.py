import pandas as pd
import numpy as np
import logging
from datetime import datetime
from catboost import CatBoostClassifier
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
        logging.FileHandler("advanced_catboost.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_advanced_catboost():
    """Train an advanced CatBoost model with enhanced features."""
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
    
    # Step 6: Convert target to binary (for classification)
    logger.info("Converting target to binary")
    y_train_binary = (y_train > 0.5).astype(int)
    y_val_binary = (y_val > 0.5).astype(int)
    
    # Step 7: Identify categorical features
    logger.info("Identifying categorical features")
    categorical_features = []
    # Convert numeric categorical features to strings to avoid CatBoost errors
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or col.startswith('humidity_') or col.startswith('cloud_'):
            categorical_features.append(col)
            # Convert numeric categorical features to strings
            if X_train[col].dtype != 'object':
                logger.info(f"Converting numeric categorical feature to string: {col}")
                X_train[col] = X_train[col].astype(str)
                X_val[col] = X_val[col].astype(str)
                test_df_cleaned[col] = test_df_cleaned[col].astype(str) if col in test_df_cleaned.columns else None
    
    logger.info(f"Found {len(categorical_features)} categorical features")
    
    # Step 8: Train CatBoost with best parameters from previous optimization
    logger.info("Training CatBoost model with best parameters")
    
    # Use your best parameters from previous CatBoost hyperparameter tuning
    best_params = {
        'iterations': 119, 
        'depth': 3, 
        'learning_rate': 0.0793818386577882, 
        'l2_leaf_reg': 0.5996537623330089, 
        'random_strength': 7.553592434650045, 
        'bagging_temperature': 3.0810865666581937, 
        'subsample': 0.6638931889134149, 
        'colsample_bylevel': 0.6722947237063621,
        'random_seed': 42
    }
    
    # Add categorical feature indices
    cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_features 
                           if col in X_train.columns]
    
    # Train on cross-validation to get a more robust model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in kf.split(X_train):
        # Split data
        cv_X_train = X_train.iloc[train_idx]
        cv_y_train = y_train_binary.iloc[train_idx]
        cv_X_test = X_train.iloc[test_idx]
        cv_y_test = y_train_binary.iloc[test_idx]
        
        # Train model
        model = CatBoostClassifier(
            **best_params,
            cat_features=cat_features_indices,
            verbose=0
        )
        
        model.fit(cv_X_train, cv_y_train)
        
        # Evaluate
        cv_preds = model.predict_proba(cv_X_test)[:, 1]
        cv_score = roc_auc_score(cv_y_test, cv_preds)
        cv_scores.append(cv_score)
    
    logger.info(f"Cross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train final validation model
    logger.info("Training final validation model")
    final_val_model = CatBoostClassifier(
        **best_params,
        cat_features=cat_features_indices,
        verbose=0
    )
    
    final_val_model.fit(X_train, y_train_binary)
    
    # Evaluate on validation data
    val_preds = final_val_model.predict_proba(X_val)[:, 1]
    val_score = roc_auc_score(y_val_binary, val_preds)
    logger.info(f"Validation AUC: {val_score:.4f}")
    
    # Step 9: Train on all data
    logger.info("Training on all data for final prediction")
    all_X = pd.concat([X_train, X_val])
    all_y = pd.concat([y_train, y_val])
    all_y_binary = (all_y > 0.5).astype(int)
    
    # Get all categorical feature indices
    all_cat_features_indices = [all_X.columns.get_loc(col) for col in categorical_features 
                               if col in all_X.columns]
    
    # Train final model
    final_model = CatBoostClassifier(
        **best_params,
        cat_features=all_cat_features_indices,
        verbose=0
    )
    
    logger.info(f"Training final model with shape: {all_X.shape}")
    final_model.fit(all_X, all_y_binary)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model for later use
    import os
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model_path = f"../models/catboost_model_{timestamp}.cbm"
    logger.info(f"Saving model to {model_path}")
    final_model.save_model(model_path)
    
    # Step 10: Make predictions on test data
    logger.info("Making predictions with final model")
    test_preds = final_model.predict_proba(test_df_cleaned.drop(columns=['id']))[:, 1]
    
    # Step 11: Create submission file
    submission_file = f"../submissions/advanced_catboost_{timestamp}.csv"
    
    create_submission(test_df_cleaned['id'], test_preds, submission_file)
    
    logger.info(f"Submission file created: {submission_file}")
    
    print(f"\nAdvanced CatBoost model submission file created: {submission_file}")
    print("\nTo submit to Kaggle, use this command:")
    print(f"kaggle competitions submit -c playground-series-s5e3 -f {submission_file} -m \"Advanced CatBoost model with enhanced features\"")
    
    return submission_file

def analyze_feature_importance(model, feature_names, train_data=None, cat_features=None):
    """Extract and display feature importance from a trained CatBoost model."""
    # Get feature importance
    if train_data is not None:
        # Convert DataFrame to CatBoost Pool if needed
        from catboost import Pool
        if not isinstance(train_data, Pool):
            try:
                # Create a Pool from DataFrame
                pool = Pool(train_data, cat_features=cat_features)
                importance = model.get_feature_importance(data=pool)
            except Exception as e:
                print(f"Error creating Pool or getting feature importance: {e}")
                print("Attempting to get feature importance without data...")
                try:
                    importance = model.get_feature_importance()
                except Exception as e2:
                    print(f"Error getting feature importance without data: {e2}")
                    print("Cannot calculate feature importance.")
                    return None
        else:
            importance = model.get_feature_importance(data=train_data)
    else:
        # Try without data (may work if model has sufficient metadata)
        try:
            importance = model.get_feature_importance()
        except Exception as e:
            print(f"Error getting feature importance without data: {e}")
            print("Cannot calculate feature importance without proper training data.")
            return None
    
    # Create a DataFrame for easier viewing
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(
        by='Importance', ascending=False
    ).reset_index(drop=True)
    
    # Print the top 20 features
    print("\nTop 20 Feature Importance:")
    for i, row in feature_importance_df.head(20).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"../feature_importance_{timestamp}.csv"
    feature_importance_df.to_csv(csv_file, index=False)
    print(f"\nComplete feature importance saved to: {csv_file}")
    
    return feature_importance_df

if __name__ == "__main__":
    submission_file = train_advanced_catboost()
    
    # Load the final model
    logger = logging.getLogger(__name__)
    logger.info("Loading the trained model for feature importance analysis")
    
    # Recreate the final model path from the submission file path
    import os
    timestamp = os.path.basename(submission_file).split('_')[-1].split('.')[0]
    
    # Load the train and test data again so we can get all_X for feature names
    logger.info("Reloading data for feature importance analysis")
    train_df, test_df = load_data('../data/train.csv', '../data/test.csv')
    train_df = create_features(train_df)
    train_df = enhanced_feature_engineering(train_df, test_df)[0]
    train_df_cleaned = clean_data(train_df, is_train=True)
    X_train, X_val, y_train, y_val = split_data(train_df_cleaned)
    all_X = pd.concat([X_train, X_val])
    
    # Load the model we just trained
    best_params = {
        'iterations': 119, 
        'depth': 3, 
        'learning_rate': 0.0793818386577882, 
        'l2_leaf_reg': 0.5996537623330089, 
        'random_strength': 7.553592434650045, 
        'bagging_temperature': 3.0810865666581937, 
        'subsample': 0.6638931889134149, 
        'colsample_bylevel': 0.6722947237063621,
        'random_seed': 42
    }
    
    # Identify categorical features again
    categorical_features = []
    for col in all_X.columns:
        if all_X[col].dtype == 'object' or col.startswith('humidity_') or col.startswith('cloud_'):
            categorical_features.append(col)
            # Convert numeric categorical features to strings
            if all_X[col].dtype != 'object':
                logger.info(f"Converting numeric categorical feature to string for analysis: {col}")
                all_X[col] = all_X[col].astype(str)
    
    # Get categorical feature indices
    all_cat_features_indices = [all_X.columns.get_loc(col) for col in categorical_features if col in all_X.columns]
    
    # Recreate the model with the same parameters
    final_model = CatBoostClassifier(
        **best_params,
        cat_features=all_cat_features_indices,
        verbose=0
    )
    
    # Load the trained model from disk if available, otherwise use the one in memory
    try:
        from catboost import CatBoostClassifier
        model_path = f"../models/catboost_model_{timestamp}.cbm"
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            final_model.load_model(model_path)
        else:
            logger.info("Using model from memory")
            # Recreate all_y_binary
            all_y = pd.concat([y_train, y_val])
            all_y_binary = (all_y > 0.5).astype(int)
            
            # Train the model again if needed (this is just a fallback)
            final_model.fit(all_X, all_y_binary)
    except Exception as e:
        logger.warning(f"Error loading model, using in-memory model: {e}")
    
    # Analyze feature importance
    logger.info("Analyzing feature importance")
    feature_importance = analyze_feature_importance(
        final_model, 
        all_X.columns.tolist(),
        train_data=all_X
    )