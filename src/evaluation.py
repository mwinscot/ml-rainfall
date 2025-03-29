from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model, X, y, cv=5):
    """Evaluate model with cross-validation."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate RMSE for each fold
    rmse_scores = -cross_val_score(model, X, y, 
                                 scoring='neg_root_mean_squared_error', 
                                 cv=kf)
    
    print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    
    return rmse_scores

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 10))
        plt.title('Feature Importance')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.show()
    else:
        print("Model doesn't support feature importance")