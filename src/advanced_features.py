import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_features.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_advanced_weather_features(df):
    """Create specialized weather-related features."""
    logger.info("Creating advanced weather features")
    df_new = df.copy()
    
    # Temperature-humidity interaction (heat index approximation)
    if all(col in df.columns for col in ['temperature', 'humidity']):
        df_new['temp_humidity_index'] = df_new['temperature'] * df_new['humidity'] / 100
        logger.info("Created temp_humidity_index feature")
    
    # Dew point depression (difference between temp and dewpoint)
    if all(col in df.columns for col in ['temperature', 'dewpoint']):
        df_new['dewpoint_depression'] = df_new['temperature'] - df_new['dewpoint']
        logger.info("Created dewpoint_depression feature")
    
    # Wind chill factor
    if all(col in df.columns for col in ['temperature', 'windspeed']):
        # Simple approximation
        df_new['wind_chill'] = 13.12 + 0.6215*df_new['temperature'] - 11.37*(df_new['windspeed']**0.16) + 0.3965*df_new['temperature']*(df_new['windspeed']**0.16)
        logger.info("Created wind_chill feature")
    
    # Vapor pressure (approximation)
    if 'dewpoint' in df.columns:
        df_new['vapor_pressure'] = 6.11 * 10**(7.5 * df_new['dewpoint'] / (237.3 + df_new['dewpoint']))
        logger.info("Created vapor_pressure feature")
    
    # Potential rainfall indicator
    if all(col in df.columns for col in ['humidity', 'cloud']):
        df_new['rainfall_potential'] = df_new['humidity'] * df_new['cloud'] / 100
        logger.info("Created rainfall_potential feature")
    
    # Diurnal temperature range (if available)
    if all(col in df.columns for col in ['maxtemp', 'mintemp']):
        df_new['temp_range'] = df_new['maxtemp'] - df_new['mintemp']
        logger.info("Created temp_range feature")
    
    # Count new features created
    new_features = set(df_new.columns) - set(df.columns)
    logger.info(f"Created {len(new_features)} new weather features: {new_features}")
    
    return df_new

def create_polynomial_interactions(df, degree=2):
    """Create polynomial features for key weather variables."""
    logger.info(f"Creating polynomial features with degree {degree}")
    
    # Select important columns that might have non-linear relationships
    important_cols = ['temperature', 'humidity', 'pressure', 'windspeed', 'cloud']
    available_cols = [col for col in important_cols if col in df.columns]
    
    if not available_cols:
        logger.warning("No important columns found for polynomial features")
        return df
    
    # Select the data
    data = df[available_cols].copy()
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(data)
    
    # Create a DataFrame with the new features
    feature_names = poly.get_feature_names_out(available_cols)
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Drop the original columns to avoid duplication
    poly_df = poly_df.drop(columns=available_cols, errors='ignore')
    
    # Concatenate with original DataFrame
    result = pd.concat([df, poly_df], axis=1)
    
    # Count new features created
    new_features = set(result.columns) - set(df.columns)
    logger.info(f"Created {len(new_features)} polynomial features")
    
    return result

def create_weather_event_indicators(df):
    """Create binary indicators for specific weather conditions."""
    logger.info("Creating weather event indicators")
    df_new = df.copy()
    
    # High humidity indicator (potential for precipitation)
    if 'humidity' in df.columns:
        df_new['high_humidity'] = (df_new['humidity'] > 80).astype(int)
    
    # Heavy cloud cover
    if 'cloud' in df.columns:
        df_new['heavy_cloud'] = (df_new['cloud'] > 70).astype(int)
    
    # Strong wind
    if 'windspeed' in df.columns:
        df_new['strong_wind'] = (df_new['windspeed'] > df_new['windspeed'].quantile(0.75)).astype(int)
    
    # Pressure drop (indicates potential for storms)
    if 'pressure' in df.columns:
        # Calculate the difference from standard pressure (1013.25 hPa)
        df_new['pressure_drop'] = (df_new['pressure'] < 1010).astype(int)
    
    # Temperature-dewpoint convergence (indicates potential precipitation)
    if all(col in df.columns for col in ['temperature', 'dewpoint']):
        df_new['temp_dewpoint_close'] = ((df_new['temperature'] - df_new['dewpoint']) < 2.5).astype(int)
    
    # Count new features created
    new_features = set(df_new.columns) - set(df.columns)
    logger.info(f"Created {len(new_features)} weather event indicators: {new_features}")
    
    return df_new

def bin_numerical_features(df):
    """Bin numerical features into categories."""
    logger.info("Binning numerical features")
    df_new = df.copy()
    
    # Bin humidity into categories
    if 'humidity' in df.columns:
        bins = [0, 30, 60, 80, 100]
        labels = ['very_low', 'low', 'moderate', 'high']
        df_new['humidity_binned'] = pd.cut(df_new['humidity'], bins=bins, labels=labels)
        # Convert to one-hot encoding
        humidity_dummies = pd.get_dummies(df_new['humidity_binned'], prefix='humidity')
        df_new = pd.concat([df_new, humidity_dummies], axis=1)
        df_new = df_new.drop('humidity_binned', axis=1)
    
    # Bin cloud cover
    if 'cloud' in df.columns:
        bins = [0, 25, 50, 75, 100]
        labels = ['clear', 'partly_cloudy', 'mostly_cloudy', 'overcast']
        df_new['cloud_binned'] = pd.cut(df_new['cloud'], bins=bins, labels=labels)
        # Convert to one-hot encoding
        cloud_dummies = pd.get_dummies(df_new['cloud_binned'], prefix='cloud')
        df_new = pd.concat([df_new, cloud_dummies], axis=1)
        df_new = df_new.drop('cloud_binned', axis=1)
    
    # Count new features created
    new_features = set(df_new.columns) - set(df.columns)
    logger.info(f"Created {len(new_features)} binned features")
    
    return df_new

def select_important_features(X_train, y_train, X_val=None, X_test=None, threshold=0.01):
    """Select important features based on model importance."""
    logger.info(f"Selecting important features with threshold {threshold}")
    
    # Avoid circular import
    import xgboost as xgb
    
    # Train a model for feature importance
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        objective='binary:logistic',
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Select features
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    
    # Transform data
    X_train_selected = pd.DataFrame(
        selector.transform(X_train),
        columns=X_train.columns[selector.get_support()],
        index=X_train.index
    )
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    logger.info(f"Top 10 features: {selected_features[:10].tolist()}")
    
    results = [X_train_selected]
    
    # Transform validation and test data if provided
    if X_val is not None:
        X_val_selected = pd.DataFrame(
            selector.transform(X_val),
            columns=X_train.columns[selector.get_support()],
            index=X_val.index
        )
        results.append(X_val_selected)
    
    if X_test is not None:
        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            columns=X_train.columns[selector.get_support()],
            index=X_test.index
        )
        results.append(X_test_selected)
    
    results.append(selected_features)
    
    return results

def enhanced_feature_engineering(train_df, test_df):
    """Apply all feature engineering steps to train and test data."""
    logger.info("Starting enhanced feature engineering")
    
    # 1. Add basic weather features
    logger.info("Creating weather-specific features...")
    train_df = create_advanced_weather_features(train_df)
    test_df = create_advanced_weather_features(test_df)
    
    # 2. Add weather event indicators
    logger.info("Creating weather event indicators...")
    train_df = create_weather_event_indicators(train_df)
    test_df = create_weather_event_indicators(test_df)
    
    # 3. Bin numerical features
    logger.info("Binning numerical features...")
    train_df = bin_numerical_features(train_df)
    test_df = bin_numerical_features(test_df)
    
    # 4. Add polynomial and interaction features - this can significantly increase feature count
    logger.info("Creating polynomial features...")
    train_df = create_polynomial_interactions(train_df, degree=2)
    test_df = create_polynomial_interactions(test_df, degree=2)
    
    # Return engineered datasets
    logger.info(f"Enhanced feature engineering complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df