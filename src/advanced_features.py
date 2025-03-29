import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_features.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_polynomial_features(X_train, X_val=None, X_test=None, degree=2):
    """Create polynomial features."""
    logger.info(f"Creating polynomial features with degree {degree}")
    
    # Initialize the transformer
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Fit and transform training data
    X_train_poly = pd.DataFrame(
        poly.fit_transform(X_train), 
        columns=poly.get_feature_names_out(X_train.columns),
        index=X_train.index
    )
    
    # Transform validation data if provided
    if X_val is not None:
        X_val_poly = pd.DataFrame(
            poly.transform(X_val),
            columns=poly.get_feature_names_out(X_val.columns),
            index=X_val.index
        )
    else:
        X_val_poly = None
    
    # Transform test data if provided
    if X_test is not None:
        X_test_poly = pd.DataFrame(
            poly.transform(X_test),
            columns=poly.get_feature_names_out(X_test.columns),
            index=X_test.index
        )
    else:
        X_test_poly = None
    
    logger.info(f"Created {X_train_poly.shape[1]} polynomial features")
    
    if X_val is not None and X_test is not None:
        return X_train_poly, X_val_poly, X_test_poly
    elif X_val is not None:
        return X_train_poly, X_val_poly
    elif X_test is not None:
        return X_train_poly, X_test_poly
    else:
        return X_train_poly

def create_pca_features(X_train, X_val=None, X_test=None, n_components=5):
    """Create PCA features."""
    logger.info(f"Creating PCA features with {n_components} components")
    
    # Initialize the transformer
    pca = PCA(n_components=n_components)
    
    # Fit and transform training data
    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train),
        columns=[f'pca_{i+1}' for i in range(n_components)],
        index=X_train.index
    )
    
    # Transform validation data if provided
    if X_val is not None:
        X_val_pca = pd.DataFrame(
            pca.transform(X_val),
            columns=[f'pca_{i+1}' for i in range(n_components)],
            index=X_val.index
        )
    else:
        X_val_pca = None
    
    # Transform test data if provided
    if X_test is not None:
        X_test_pca = pd.DataFrame(
            pca.transform(X_test),
            columns=[f'pca_{i+1}' for i in range(n_components)],
            index=X_test.index
        )
    else:
        X_test_pca = None
    
    # Log variance explained
    logger.info(f"Variance explained by components: {pca.explained_variance_ratio_}")
    logger.info(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    if X_val is not None and X_test is not None:
        return X_train_pca, X_val_pca, X_test_pca, pca
    elif X_val is not None:
        return X_train_pca, X_val_pca, pca
    elif X_test is not None:
        return X_train_pca, X_test_pca, pca
    else:
        return X_train_pca, pca

def create_cluster_features(X_train, X_val=None, X_test=None, n_clusters=5):
    """Create cluster membership features."""
    logger.info(f"Creating cluster features with {n_clusters} clusters")
    
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit and predict on training data
    X_train_cluster = pd.DataFrame(
        {'cluster': kmeans.fit_predict(X_train)},
        index=X_train.index
    )
    
    # One-hot encode the cluster labels
    X_train_cluster = pd.get_dummies(X_train_cluster, columns=['cluster'], prefix='cluster')
    
    # Predict on validation data if provided
    if X_val is not None:
        X_val_cluster = pd.DataFrame(
            {'cluster': kmeans.predict(X_val)},
            index=X_val.index
        )
        X_val_cluster = pd.get_dummies(X_val_cluster, columns=['cluster'], prefix='cluster')
        
        # Ensure all clusters are represented in validation set
        for i in range(n_clusters):
            if f'cluster_{i}' not in X_val_cluster.columns:
                X_val_cluster[f'cluster_{i}'] = 0
    else:
        X_val_cluster = None
    
    # Predict on test data if provided
    if X_test is not None:
        X_test_cluster = pd.DataFrame(
            {'cluster': kmeans.predict(X_test)},
            index=X_test.index
        )
        X_test_cluster = pd.get_dummies(X_test_cluster, columns=['cluster'], prefix='cluster')
        
        # Ensure all clusters are represented in test set
        for i in range(n_clusters):
            if f'cluster_{i}' not in X_test_cluster.columns:
                X_test_cluster[f'cluster_{i}'] = 0
    else:
        X_test_cluster = None
    
    logger.info(f"Created {X_train_cluster.shape[1]} cluster features")
    
    if X_val is not None and X_test is not None:
        return X_train_cluster, X_val_cluster, X_test_cluster, kmeans
    elif X_val is not None:
        return X_train_cluster, X_val_cluster, kmeans
    elif X_test is not None:
        return X_train_cluster, X_test_cluster, kmeans
    else:
        return X_train_cluster, kmeans

def add_advanced_features(X_train, X_val=None, X_test=None):
    """Create all advanced features and combine them."""
    logger.info("Creating advanced features")
    
    # Keep only a subset of original features to reduce dimensionality for polynomial features
    important_features = ['temperature', 'humidity', 'pressure', 'windspeed', 'cloud']
    X_train_subset = X_train[important_features].copy() if all(col in X_train.columns for col in important_features) else X_train.copy()
    
    if X_val is not None:
        X_val_subset = X_val[important_features].copy() if all(col in X_val.columns for col in important_features) else X_val.copy()
    else:
        X_val_subset = None
        
    if X_test is not None:
        X_test_subset = X_test[important_features].copy() if all(col in X_test.columns for col in important_features) else X_test.copy()
    else:
        X_test_subset = None
    
    # Create polynomial features
    logger.info("Creating polynomial features")
    X_train_poly, X_val_poly, X_test_poly = create_polynomial_features(
        X_train_subset, X_val_subset, X_test_subset, degree=2
    ) if X_val is not None and X_test is not None else (None, None, None)
    
    # Create PCA features
    logger.info("Creating PCA features")
    X_train_pca, X_val_pca, X_test_pca, _ = create_pca_features(
        X_train, X_val, X_test, n_components=min(5, X_train.shape[1]-1)
    ) if X_val is not None and X_test is not None else (None, None, None, None)
    
    # Create cluster features
    logger.info("Creating cluster features")
    X_train_cluster, X_val_cluster, X_test_cluster, _ = create_cluster_features(
        X_train, X_val, X_test, n_clusters=5
    ) if X_val is not None and X_test is not None else (None, None, None, None)
    
    # Combine all features
    logger.info("Combining all features")
    X_train_advanced = pd.concat([X_train, X_train_poly, X_train_pca, X_train_cluster], axis=1)
    
    if X_val is not None and X_test is not None:
        X_val_advanced = pd.concat([X_val, X_val_poly, X_val_pca, X_val_cluster], axis=1)
        X_test_advanced = pd.concat([X_test, X_test_poly, X_test_pca, X_test_cluster], axis=1)
        
        logger.info(f"Final feature counts - Train: {X_train_advanced.shape[1]}, Val: {X_val_advanced.shape[1]}, Test: {X_test_advanced.shape[1]}")
        return X_train_advanced, X_val_advanced, X_test_advanced
    
    elif X_val is not None:
        X_val_advanced = pd.concat([X_val, X_val_poly, X_val_pca, X_val_cluster], axis=1)
        
        logger