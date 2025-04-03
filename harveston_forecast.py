import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ----------------------------
# 1. Data Loading & Preprocessing
# ----------------------------
def load_data():
    """Load and validate data with robust error handling"""
    try:
        # Use the correct path to data files in the data subdirectory
        train_path = os.path.join("data", "train.csv")
        train = pd.read_csv(train_path)
        print("✅ Data loaded successfully")
        print(f"📊 Dataset shape: {train.shape}")
        print("📋 Columns:", train.columns.tolist())
        
        # Check for test data
        try:
            test_path = os.path.join("data", "test.csv")
            test = pd.read_csv(test_path)
            print("✅ Test data found")
        except FileNotFoundError:
            test = None
            print("⚠️ No test.csv found - proceeding with training only")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please ensure data files exist in the 'data' directory")
        return None, None
    
    return train, test

def preprocess_data(train, test=None):
    """Preprocess data with comprehensive feature engineering and cleaning"""
    if train is None:
        return None, None
    
    # Create deep copies to avoid any modifications to the original dataframes
    train = train.copy(deep=True)
    if test is not None:
        test = test.copy(deep=True)
    
    # First, encode categorical variables immediately to avoid issues later
    categorical_cols = ['kingdom']
    encoders = {}
    
    for col in categorical_cols:
        if col in train.columns:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            encoders[col] = le
            
            # Apply same transformation to test data if available
            if test is not None and col in test.columns:
                # Handle unknown categories in test data
                test_categories = set(test[col].astype(str).unique())
                train_categories = set(le.classes_)
                unknown_categories = test_categories - train_categories
                
                if unknown_categories:
                    print(f"⚠️ Found unknown categories in test '{col}': {unknown_categories}")
                    # Map unknown categories to the most common category in training
                    most_common = train[col].mode()[0]
                    for cat in unknown_categories:
                        test.loc[test[col].astype(str) == cat, col] = most_common
                
                try:
                    test[col] = le.transform(test[col].astype(str))
                except ValueError as e:
                    print(f"⚠️ Error transforming test '{col}': {e}")
                    # Fallback: use a fresh encoder for test data
                    test[col] = LabelEncoder().fit_transform(test[col].astype(str))
    
    # Save encoders for later use
    joblib.dump(encoders, 'category_encoders.pkl')
    
    def process_df(df):
        if df is None:
            return None
            
        # Convert temperature units (handle Kelvin values)
        temp_cols = ['Avg_Temperature', 'Avg_Feels_Like_Temperature']
        for col in temp_cols:
            if col in df.columns and df[col].notna().any():
                # Identify likely Kelvin values (>100)
                kelvin_mask = df[col] > 100
                df.loc[kelvin_mask, col] = df.loc[kelvin_mask, col] - 273.15
                
        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Already encoded, so use numeric fill
                df[col].fillna(-1, inplace=True)
        
        return df
    
    train = process_df(train)
    if test is not None:
        test = process_df(test)
    
    return train, test

# ----------------------------
# 2. Feature Engineering
# ----------------------------
def create_features(train_df, test_df=None):
    """
    Creates features that can be safely applied to both train and test data
    
    Args:
        train_df (pd.DataFrame): Training DataFrame
        test_df (pd.DataFrame): Test DataFrame (optional)
    
    Returns:
        tuple: Processed training and test DataFrames
    """
    if train_df is None:
        return None, None
    
    # Make deep copies to avoid any SettingWithCopyWarning
    train_df = train_df.copy(deep=True)
    
    # Only create features that can be applied to both datasets
    
    # Time-based features
    if all(col in train_df.columns for col in ['Year', 'Month', 'Day']):
        try:
            train_df['Day_of_year'] = pd.to_datetime(
                train_df[['Year', 'Month', 'Day']].rename(columns={
                    'Year': 'year', 'Month': 'month', 'Day': 'day'
                }), errors='coerce'
            ).dt.dayofyear
            
            # Cyclical encoding for day of year
            train_df['Day_of_year_sin'] = np.sin(2 * np.pi * train_df['Day_of_year']/365)
            train_df['Day_of_year_cos'] = np.cos(2 * np.pi * train_df['Day_of_year']/365)
            
            # Month as cyclical feature
            train_df['Month_sin'] = np.sin(2 * np.pi * train_df['Month']/12)
            train_df['Month_cos'] = np.cos(2 * np.pi * train_df['Month']/12)
        except Exception as e:
            print(f"⚠️ Warning: Could not create date-based features for training: {e}")
    
    # Location-based features
    if 'latitude' in train_df.columns and 'longitude' in train_df.columns:
        train_df['lat_long_combined'] = train_df['latitude'] * train_df['longitude']
    
    # Simple interaction features
    if 'Avg_Temperature' in train_df.columns and 'Radiation' in train_df.columns:
        if not (train_df['Avg_Temperature'].isna().all() or train_df['Radiation'].isna().all()):
            train_df['temp_radiation_interaction'] = train_df['Avg_Temperature'] * train_df['Radiation']
    
    # Now process test data if available
    if test_df is not None:
        test_df = test_df.copy(deep=True)
        
        # Apply same transformations to test data
        if all(col in test_df.columns for col in ['Year', 'Month', 'Day']):
            try:
                test_df['Day_of_year'] = pd.to_datetime(
                    test_df[['Year', 'Month', 'Day']].rename(columns={
                        'Year': 'year', 'Month': 'month', 'Day': 'day'
                    }), errors='coerce'
                ).dt.dayofyear
                
                test_df['Day_of_year_sin'] = np.sin(2 * np.pi * test_df['Day_of_year']/365)
                test_df['Day_of_year_cos'] = np.cos(2 * np.pi * test_df['Day_of_year']/365)
                
                test_df['Month_sin'] = np.sin(2 * np.pi * test_df['Month']/12)
                test_df['Month_cos'] = np.cos(2 * np.pi * test_df['Month']/12)
            except Exception as e:
                print(f"⚠️ Warning: Could not create date-based features for test: {e}")
        
        # Location-based features
        if 'latitude' in test_df.columns and 'longitude' in test_df.columns:
            test_df['lat_long_combined'] = test_df['latitude'] * test_df['longitude']
        
        # Simple interaction features
        if 'Avg_Temperature' in test_df.columns and 'Radiation' in test_df.columns:
            if not (test_df['Avg_Temperature'].isna().all() or test_df['Radiation'].isna().all()):
                test_df['temp_radiation_interaction'] = test_df['Avg_Temperature'] * test_df['Radiation']
    
    return train_df, test_df

# ----------------------------
# 3. Model Training
# ----------------------------
def train_xgboost(X_train, y_train):
    """Train XGBoost model with optimized parameters"""
    print("\n🏗 Training XGBoost model...")
    
    # Check if we have data to train on
    if X_train is None or y_train is None or X_train.shape[0] == 0 or y_train.shape[0] == 0:
        print("❌ No data available for training XGBoost model")
        return None
    
    # Save the column names for later reference
    joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
    
    try:
        model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            ),
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("✅ XGBoost training complete")
        return model
    except Exception as e:
        print(f"❌ Error training XGBoost model: {e}")
        return None

def train_lstm(X_train, y_train):
    """Train LSTM model with early stopping and gradient clipping"""
    print("\n🧠 Training LSTM model...")
    
    # Check if we have data to train on
    if X_train is None or y_train is None or X_train.shape[0] == 0 or y_train.shape[0] == 0:
        print("❌ No data available for training LSTM model")
        return None, None, None
    
    try:
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_train)
        
        # Also scale the target to prevent exploding gradients
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_train)
        
        # Save both scalers for inference
        joblib.dump(scaler_X, 'scaler_X.pkl')
        joblib.dump(scaler_y, 'scaler_y.pkl')
        
        # Check for any remaining NaN values
        if np.isnan(X_scaled).any() or np.isnan(y_scaled).any():
            print("⚠️ Warning: NaN values found after scaling. Replacing with zeros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            y_scaled = np.nan_to_num(y_scaled, nan=0.0)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Use a smaller network and add regularization
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(1, X_scaled.shape[1]),
                 kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
            Dropout(0.3),
            LSTM(32, kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(y_train.shape[1])
        ])
        
        # Use gradient clipping to prevent exploding gradients
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse') # Change to MSE from MAE
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            min_delta=0.001 # Small improvement threshold
        )
        
        history = model.fit(
            X_reshaped, y_scaled, # Use scaled y values
            epochs=100, # Reduce epochs
            batch_size=32, # Smaller batch size
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        print("✅ LSTM training complete")
        return model, history, scaler_y # Return the y scaler too
    except Exception as e:
        print(f"❌ Error training LSTM model: {e}")
        return None, None, None

# ----------------------------
# 4. Evaluation Metrics
# ----------------------------
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def evaluate_model(model, X_val, y_val, model_type='XGBoost', scaler_y=None):
    """Evaluate model and return metrics"""
    print(f"\n📊 Evaluating {model_type} model...")
    
    # Check if we have a model and validation data
    if model is None or X_val is None or y_val is None:
        print(f"❌ Cannot evaluate {model_type} model: Missing model or validation data")
        return None, float('inf')
    
    try:
        if model_type == 'LSTM':
            # Scale features using saved scaler
            try:
                scaler_X = joblib.load('scaler_X.pkl')
                X_val_scaled = scaler_X.transform(X_val)
                
                # Check for NaN values
                if np.isnan(X_val_scaled).any():
                    print("⚠️ Warning: NaN values found in validation data. Replacing with zeros.")
                    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
                
                X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
                y_pred_scaled = model.predict(X_val_reshaped)
                
                # Inverse transform predictions back to original scale
                if scaler_y is not None:
                    y_pred = scaler_y.inverse_transform(y_pred_scaled)
                else:
                    print("⚠️ No y scaler found, using scaled predictions")
                    y_pred = y_pred_scaled
            except Exception as e:
                print(f"❌ Error during LSTM prediction: {e}")
                return None, float('inf')
        else:
            y_pred = model.predict(X_val)
        
        scores = {}
        for i, col in enumerate(y_val.columns):
            # Convert to numpy arrays to make slicing easier
            y_val_np = y_val[col].to_numpy()
            y_pred_np = y_pred[:, i]
            
            # Find indices where both arrays have valid values
            valid_mask = ~(np.isnan(y_val_np) | np.isnan(y_pred_np))
            
            if np.sum(valid_mask) > 0:
                smape_score = smape(y_val_np[valid_mask], y_pred_np[valid_mask])
                mae = mean_absolute_error(y_val_np[valid_mask], y_pred_np[valid_mask])
                scores[col] = {'SMAPE': smape_score, 'MAE': mae}
                print(f"{col}: SMAPE = {smape_score:.2f}%, MAE = {mae:.2f}")
            else:
                print(f"⚠️ No valid data points for {col}, skipping metrics")
                scores[col] = {'SMAPE': float('inf'), 'MAE': float('inf')}
        
        # Only average valid scores
        valid_scores = [v['SMAPE'] for v in scores.values() if v['SMAPE'] != float('inf')]
        if valid_scores:
            avg_smape = np.mean(valid_scores)
            print(f"Average SMAPE: {avg_smape:.2f}%")
        else:
            avg_smape = float('inf')
            print("⚠️ No valid SMAPE scores to average")
        
        return scores, avg_smape
    except Exception as e:
        print(f"❌ Error evaluating {model_type} model: {e}")
        return None, float('inf')

# ----------------------------
# 5. Prediction Pipeline
# ----------------------------
def align_test_features(test_df, feature_cols):
    """Ensure test data has the same features as training data"""
    test_df = test_df.copy()
    
    # Add missing columns
    for col in feature_cols:
        if col not in test_df.columns:
            print(f"⚠️ Adding missing column to test: {col}")
            test_df[col] = 0
    
    # Select only the relevant columns and in the right order
    test_df = test_df[feature_cols]
    
    return test_df

def predict_and_save(model, test_df, model_name, scaler_y=None):
    """Generate predictions and save submission file"""
    print(f"\n🔮 Generating {model_name} predictions...")
    
    # Check if we have a model and test data
    if model is None or test_df is None:
        print(f"❌ Cannot generate {model_name} predictions: Missing model or test data")
        return None
    
    try:
        # Get the feature columns used during training
        feature_cols = joblib.load('feature_columns.pkl')
        
        # Align test features with training features
        X_test = align_test_features(test_df, feature_cols)
        
        if model_name == 'LSTM':
            # Scale features using saved scaler
            try:
                scaler_X = joblib.load('scaler_X.pkl')
                X_test_scaled = scaler_X.transform(X_test)
                
                # Check for NaN values
                if np.isnan(X_test_scaled).any():
                    print("⚠️ Warning: NaN values found in test data. Replacing with zeros.")
                    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
                
                X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                y_pred_scaled = model.predict(X_test_reshaped)
                
                # Inverse transform predictions back to original scale
                if scaler_y is not None:
                    y_pred = scaler_y.inverse_transform(y_pred_scaled)
                else:
                    print("⚠️ No y scaler found, using scaled predictions")
                    y_pred = y_pred_scaled
            except Exception as e:
                print(f"❌ Error during LSTM prediction: {e}")
                return None
        else:
            y_pred = model.predict(X_test)
        
        # Define target columns based on problem statement
        target_cols = [
            'Avg_Temperature',
            'Radiation',
            'Rain_Amount',
            'Wind_Speed',
            'Wind_Direction'
        ]
        
        submission = pd.DataFrame({
            'ID': test_df['ID'],
        })
        
        # Add each target column
        for i, col in enumerate(target_cols):
            if i < y_pred.shape[1]:
                submission[col] = y_pred[:, i]
            else:
                print(f"⚠️ Warning: Missing prediction for {col}. Using zeros.")
                submission[col] = 0
        
        # Save the submission in the data directory with the required name
        submission_file = os.path.join("data", "sample_submission.csv")
        submission.to_csv(submission_file, index=False)
        print(f"✅ {model_name} predictions saved to {submission_file}")
        return submission
        
    except Exception as e:
        print(f"❌ Error generating {model_name} predictions: {e}")
        return None

# ----------------------------
# Visualization Utilities
# ----------------------------
def plot_training_history(history):
    """Plot LSTM training history"""
    if history is None:
        print("❌ No training history to plot")
        return
        
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        print("✅ Training history plot saved to training_history.png")
    except Exception as e:
        print(f"❌ Error plotting training history: {e}")

def plot_kingdom_features(train_df):
    """Plot features by kingdom to understand regional patterns"""
    if 'kingdom' not in train_df.columns:
        print("❌ No kingdom column found for regional analysis")
        return
        
    try:
        # Use the original kingdom values for better visualization
        encoders = joblib.load('category_encoders.pkl')
        kingdom_encoder = encoders.get('kingdom')
        
        if kingdom_encoder:
            kingdom_names = kingdom_encoder.classes_
            kingdom_ids = list(range(len(kingdom_names)))
            
            target_cols = [
                'Avg_Temperature', 
                'Radiation', 
                'Rain_Amount', 
                'Wind_Speed', 
                'Wind_Direction'
            ]
            
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(target_cols):
                plt.subplot(len(target_cols), 1, i+1)
                for k_id, k_name in zip(kingdom_ids, kingdom_names):
                    kingdom_data = train_df[train_df['kingdom'] == k_id]
                    if 'Month' in kingdom_data.columns and col in kingdom_data.columns:
                        avg_by_month = kingdom_data.groupby('Month')[col].mean()
                        plt.plot(avg_by_month.index, avg_by_month.values, marker='o', label=k_name)
                plt.title(f'{col} by Month Across Kingdoms')
                plt.ylabel(col)
                if i == 0: # Only show legend on first subplot
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig('kingdom_patterns.png')
            plt.close()
            print("✅ Kingdom feature patterns saved to kingdom_patterns.png")
        else:
            print("⚠️ Kingdom encoder not found, skipping kingdom visualization")
    except Exception as e:
        print(f"❌ Error plotting kingdom features: {e}")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting Harveston Climate Prediction Pipeline")
    
    # 1. Load and preprocess data
    print("\n🔍 Loading and preprocessing data...")
    train, test = load_data()
    
    # Check if we have data to work with
    if train is None:
        print("❌ No training data available. Exiting.")
        import sys
        sys.exit(1)
        
    # Define target columns based on the problem statement
    target_cols = [
        'Avg_Temperature',
        'Radiation',
        'Rain_Amount',
        'Wind_Speed',
        'Wind_Direction'
    ]
    
    # Preprocess data (including encoding categorical variables)
    train, test = preprocess_data(train, test)
    
    # Plot key patterns by kingdom
    plot_kingdom_features(train)
    
    # 2. Feature engineering (apply to both train and test)
    print("\n⚙ Creating features...")
    train, test = create_features(train, test)
    
    # Prepare data for training
    X_columns = [col for col in train.columns if col not in ['ID'] + target_cols]
    X = train[X_columns]
    y = train[target_cols]
    
    # 3. Train-test split (time-series aware)
    print("\n✂️ Creating time-series validation split...")
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_index, val_index = splits[-1] # Use the last split
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    except Exception as e:
        print(f"⚠️ Warning: Time series split failed: {e}. Using random split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train models
    xgb_model = train_xgboost(X_train, y_train)
    lstm_model, history, scaler_y = train_lstm(X_train, y_train)
    
    # 5. Evaluate models
    print("\n📈 Model Evaluation Results:")
    print("--------------------------")
    
    models = {}
    model_scores = {}
    
    if xgb_model is not None:
        xgb_scores, xgb_avg = evaluate_model(xgb_model, X_val, y_val, 'XGBoost')
        models['XGBoost'] = xgb_model
        model_scores['XGBoost'] = xgb_avg
    
    if lstm_model is not None:
        lstm_scores, lstm_avg = evaluate_model(lstm_model, X_val, y_val, 'LSTM', scaler_y)
        models['LSTM'] = lstm_model
        model_scores['LSTM'] = lstm_avg
        
        # Plot training history
        plot_training_history(history)
    
    # 6. Generate predictions if test data exists
    if test is not None:
        print("\n📊 Generating predictions...")
        
        submissions = {}
        
        if xgb_model is not None:
            xgb_submission = predict_and_save(xgb_model, test, "XGBoost")
            if xgb_submission is not None:
                submissions['XGBoost'] = xgb_submission
                
        if lstm_model is not None:
            lstm_submission = predict_and_save(lstm_model, test, "LSTM", scaler_y)
            if lstm_submission is not None:
                submissions['LSTM'] = lstm_submission
        
        # Create ensemble prediction (weighted average) if both models available
        if len(submissions) >= 2:
            print("\n🤝 Creating ensemble predictions...")
            try:
                # Select the best performing model for each target
                best_model = min(model_scores, key=model_scores.get)
                print(f"🏆 Best overall model: {best_model} (SMAPE: {model_scores[best_model]:.2f}%)")
                
                # Create weighted ensemble (inversely weighted by SMAPE score)
                total_weight = sum(1/score for score in model_scores.values() if score != float('inf'))
                weights = {model: (1/score)/total_weight if score != float('inf') else 0 
                           for model, score in model_scores.items()}
                
                print("⚖️ Model weights for ensemble:")
                for model, weight in weights.items():
                    print(f" - {model}: {weight:.2f}")
                
                # Initialize ensemble DataFrame
                ensemble_submission = pd.DataFrame({'ID': test['ID']})
                
                # Calculate weighted predictions
                for col in target_cols:
                    ensemble_submission[col] = 0
                    for model, weight in weights.items():
                        if model in submissions and col in submissions[model].columns and weight > 0:
                            ensemble_submission[col] += weight * submissions[model][col]
                
                # Save ensemble predictions
                ensemble_file = os.path.join("data", "sample_submission.csv")
                ensemble_submission.to_csv(ensemble_file, index=False)
                print(f"✅ Ensemble predictions saved to {ensemble_file}")
                
            except Exception as e:
                print(f"❌ Error creating ensemble predictions: {e}")
                # Fall back to the best single model
                if best_model in submissions:
                    best_submission = submissions[best_model]
                    best_submission.to_csv(os.path.join("data", "sample_submission.csv"), index=False)
                    print(f"✅ Falling back to best model ({best_model}) predictions")
    
    print("\n🎉 Harveston Climate Prediction Pipeline completed successfully!")