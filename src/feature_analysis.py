import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
import logging

# Import modules
from src.data_processing import load_data, clean_data, split_data, scale_features
from src.feature_engineering import create_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_features():
    # Step 1: Load and process data
    train_df, _ = load_data('data/train.csv', 'data/test.csv')
    train_df = create_features(train_df)
    train_df_cleaned = clean_data(train_df, is_train=True)
    
    # Step 2: Split data
    X_train, X_val, y_train, y_val = split_data(train_df_cleaned)
    
    # Step 3: Scale features
    X_train_scaled, X_val_scaled, _, scaler = scale_features(X_train, X_val)
    
    # Step 4: Train a model
    logger.info("Training XGBoost model for feature importance analysis")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Step 5: Get feature importance from the model
    logger.info("Calculating feature importances")
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Create a DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Step 6: Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Step 7: Calculate permutation importance
    logger.info("Calculating permutation importance")
    perm_importance = permutation_importance(
        model, X_val_scaled, y_val, 
        n_repeats=10, 
        random_state=42
    )
    
    # Create a DataFrame for easier manipulation
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    # Step 8: Plot permutation importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=perm_importance_df)
    plt.title('Permutation Importance')
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    
    # Step 9: SHAP values for feature importance
    try:
        logger.info("Calculating SHAP values")
        explainer = shap.Explainer(model)
        shap_values = explainer(X_val_scaled)
        
        # Plot SHAP values
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_val_scaled, feature_names=feature_names)
        plt.savefig('shap_summary.png')
        
        # Plot detailed SHAP values for top features
        for feature in importance_df['feature'].head(5):
            plt.figure(figsize=(10, 6))
            feature_idx = list(feature_names).index(feature)
            shap.dependence_plot(feature_idx, shap_values.values, X_val_scaled, feature_names=feature_names)
            plt.savefig(f'shap_dependence_{feature}.png')
            
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
    
    # Step 10: Print feature importance summary
    print("\nFeature Importance Summary:")
    print(importance_df.head(10))
    
    print("\nPermutation Importance Summary:")
    print(perm_importance_df.head(10))
    
    logger.info("Feature analysis complete")
    print("\nFeature importance plots saved as PNG files")

if __name__ == "__main__":
    analyze_features()