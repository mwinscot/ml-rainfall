import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(train_path, test_path):
    """
    Load training and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to the training CSV file
    test_path : str
        Path to the test CSV file
        
    Returns:
    --------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    """
    logger.info(f"Loading data from {train_path} and {test_path}")
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def inspect_data(df, name="dataset"):
    """
    Inspect dataset for basic properties and issues.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to inspect
    name : str
        Name of the dataset for logging
        
    Returns:
    --------
    stats : dict
        Dictionary containing statistics about the dataset
    """
    logger.info(f"Inspecting {name}")
    
    stats = {}
    
    # Basic info
    stats['shape'] = df.shape
    stats['columns'] = list(df.columns)
    
    # Missing values
    missing_values = df.isnull().sum()
    stats['missing_values'] = missing_values[missing_values > 0].to_dict()
    missing_percent = (missing_values / len(df)) * 100
    stats['missing_percent'] = missing_percent[missing_percent > 0].to_dict()
    
    # Data types
    stats['dtypes'] = df.dtypes.to_dict()
    
    # Numeric features statistics
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_columns.empty:
        stats['numeric_stats'] = df[numeric_columns].describe().to_dict()
    
    # Categorical features statistics
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_columns.empty:
        stats['categorical_stats'] = {col: df[col].value_counts().to_dict() for col in categorical_columns}
        stats['categorical_unique'] = {col: df[col].nunique() for col in categorical_columns}
    
    # Log some key information
    logger.info(f"{name} shape: {stats['shape']}")
    if stats.get('missing_values'):
        logger.info(f"{name} missing values: {stats['missing_values']}")
    
    return stats

def clean_data(df, is_train=True):
    """
    Clean the dataset by handling missing values, outliers, etc.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to clean
    is_train : bool
        Whether this is the training dataset
        
    Returns:
    --------
    cleaned_df : pandas.DataFrame
        Cleaned dataset
    """
    logger.info(f"Cleaning {'train' if is_train else 'test'} dataset")
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = cleaned_df.select_dtypes(include=['object', 'category']).columns
    
    # For numeric columns, use median imputation
    if not numeric_columns.empty:
        numeric_imputer = SimpleImputer(strategy='median')
        cleaned_df[numeric_columns] = numeric_imputer.fit_transform(cleaned_df[numeric_columns])
    
    # For categorical columns, use most frequent imputation
    if not categorical_columns.empty:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        cleaned_df[categorical_columns] = categorical_imputer.fit_transform(cleaned_df[categorical_columns])
    
    # Handle outliers (using IQR method for numeric columns)
    if is_train:  # Only detect outliers in training data
        for col in numeric_columns:
            if col not in ['id', 'rainfall']:  # Skip ID and target columns
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    logger.info(f"Capped {outliers_count} outliers in column {col}")
    
    logger.info(f"Cleaned {'train' if is_train else 'test'} dataset, shape: {cleaned_df.shape}")
    return cleaned_df

def split_data(df, target_col='rainfall', test_size=0.2, random_state=42):
    """
    Split the dataset into training and validation sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to split
    target_col : str
        Column name of the target variable
    test_size : float
        Proportion of the dataset to include in the validation split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_val : pandas.DataFrame
        Validation features
    y_train : pandas.Series
        Training target
    y_val : pandas.Series
        Validation target
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    X = df.drop(columns=[target_col, 'id'] if 'id' in df.columns else [target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Split data - X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def scale_features(X_train, X_val=None, X_test=None, scaler_type='standard'):
    """
    Scale numerical features.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_val : pandas.DataFrame, optional
        Validation features
    X_test : pandas.DataFrame, optional
        Test features
    scaler_type : str
        Type of scaler to use ('standard' or 'robust')
        
    Returns:
    --------
    X_train_scaled : pandas.DataFrame
        Scaled training features
    X_val_scaled : pandas.DataFrame, optional
        Scaled validation features
    X_test_scaled : pandas.DataFrame, optional
        Scaled test features
    scaler : object
        Fitted scaler object
    """
    logger.info(f"Scaling features using {scaler_type} scaler")
    
    # Select numeric columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Initialize scaler
    if scaler_type.lower() == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # Create copies to avoid modifying originals
    X_train_scaled = X_train.copy()
    
    # Fit and transform training data
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    # Transform validation data if provided
    if X_val is not None:
        X_val_scaled = X_val.copy()
        X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    else:
        X_val_scaled = None
    
    # Transform test data if provided
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    else:
        X_test_scaled = None
    
    logger.info("Feature scaling completed")
    
    if X_val is not None and X_test is not None:
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    elif X_val is not None:
        return X_train_scaled, X_val_scaled, scaler
    elif X_test is not None:
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, scaler

def get_feature_types(df):
    """
    Categorize features as numeric or categorical.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    numeric_features : list
        List of numeric feature names
    categorical_features : list
        List of categorical feature names
    """
    # Skip id and target columns
    exclude_cols = ['id', 'rainfall']
    relevant_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify feature types
    numeric_features = df[relevant_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[relevant_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Identified {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
    
    return numeric_features, categorical_features

def main():
    """Main function to execute data processing pipeline for testing."""
    # Example usage
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    # Load data
    train_df, test_df = load_data(train_path, test_path)
    
    # Inspect data
    train_stats = inspect_data(train_df, "train")
    test_stats = inspect_data(test_df, "test")
    
    # Clean data
    train_df_cleaned = clean_data(train_df, is_train=True)
    test_df_cleaned = clean_data(test_df, is_train=False)
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(train_df_cleaned)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, test_df_cleaned.drop(columns=['id'])
    )
    
    logger.info("Data processing pipeline completed successfully")
    
    # Return processed datasets for further use
    return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, test_df_cleaned['id']

if __name__ == "__main__":
    main()