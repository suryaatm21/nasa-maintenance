import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Try to import XGBoost, but continue if not available
try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    print("Warning: XGBoost not available. XGBoost models will be skipped.")
    xgboost_available = False

# Try to import TensorFlow and Keras, but continue if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tensorflow_available = True
except ImportError:
    print("Warning: TensorFlow/Keras not available. Neural network models will be skipped.")
    tensorflow_available = False

# Try to import matplotlib and seaborn, but continue if not available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plotting_available = True
except ImportError:
    print("Warning: matplotlib and/or seaborn not available. Visualizations will be skipped.")
    plotting_available = False

def detect_outliers(data, columns, n_std=3):
    """
    Detect outliers using the Z-score method.
    """
    outlier_mask = np.zeros(len(data), dtype=bool)
    for col in columns:
        z_scores = np.abs(stats.zscore(data[col]))
        outlier_mask |= z_scores > n_std
    return outlier_mask

def add_engineered_features(data):
    """Add engineered features to the dataset."""
    # Create a copy of the dataframe to avoid performance warnings
    data = data.copy()
    
    # Identify sensor columns
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    
    # Add rolling statistics (window size of 5 cycles)
    window_size = 5
    
    for col in sensor_cols:
        # Rolling mean
        rolling_mean = data.groupby('engine_id')[col].rolling(window=window_size, min_periods=1).mean().reset_index(0, drop=True)
        data[f'{col}_rolling_mean'] = rolling_mean
        
        # Rolling standard deviation
        rolling_std = data.groupby('engine_id')[col].rolling(window=window_size, min_periods=1).std().reset_index(0, drop=True)
        data[f'{col}_rolling_std'] = rolling_std.fillna(0)
        
        # Calculate rate of change
        rate = data.groupby('engine_id')[col].diff().fillna(0)
        data[f'{col}_rate'] = rate
        
        # Exponential weighted moving average
        data[f'{col}_ewma'] = data.groupby('engine_id')[col].ewm(span=5).mean().reset_index(0, drop=True)
    
    # Add time since last peak/valley (if there are enough cycles)
    for col in sensor_cols:
        try:
            peak_indices = data.groupby('engine_id').apply(
                lambda x: x[col].diff().diff().apply(lambda y: 1 if y < 0 else 0 if y > 0 else 0.5)
            ).reset_index(0, drop=True)
            data[f'{col}_peak_indicator'] = peak_indices
        except:
            # Skip if not enough data points
            pass
    
    # Feature interactions between important sensors
    # Could add more based on further analysis
    if 'sensor_2' in data.columns and 'sensor_3' in data.columns:
        data['sensor_2_3_ratio'] = data['sensor_2'] / data['sensor_3'].replace(0, 0.001)
    
    if 'sensor_4' in data.columns and 'sensor_7' in data.columns:
        data['sensor_4_7_sum'] = data['sensor_4'] + data['sensor_7']
    
    if 'sensor_11' in data.columns and 'sensor_12' in data.columns:
        data['sensor_11_12_diff'] = data['sensor_11'] - data['sensor_12']
    
    # Add time features
    data['cycle_normalized'] = data.groupby('engine_id')['time_in_cycles'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )
    
    # Add polynomial features for time
    data['cycle_squared'] = data['time_in_cycles'] ** 2
    data['cycle_cubed'] = data['time_in_cycles'] ** 3
    
    return data

def select_features(X_train, y_train, X_test, n_features=20):
    """
    Select most important features using correlation with target.
    """
    selector = SelectKBest(score_func=f_regression, k=n_features)
    
    # Fit and transform the training data
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    
    # Convert to DataFrame with proper column names
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
    
    return X_train_selected, X_test_selected

def load_and_process_data(file_path, is_train=True):
    """
    Load and process the C-MAPSS dataset.
    """
    column_names = ['engine_id', 'time_in_cycles', 'operational_setting_1',
                    'operational_setting_2', 'operational_setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    data = pd.read_csv(file_path, sep=' ', header=None, names=column_names, engine='python')
    
    data = data.dropna(axis=1, how='all')
    
    if is_train:
        max_cycle_per_engine = data.groupby('engine_id')['time_in_cycles'].max().reset_index()
        max_cycle_per_engine.columns = ['engine_id', 'max_cycle']
        
        data = data.merge(max_cycle_per_engine, on='engine_id')
        data['RUL'] = data['max_cycle'] - data['time_in_cycles']
        data = data.drop(columns=['max_cycle'])
    
    return data

def scale_features(data, scaler=None, is_train=True):
    """
    Scale sensor readings and operational settings using MinMaxScaler.
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Dynamically get feature columns that exist in the data
    operational_settings = [col for col in data.columns if col.startswith('operational_setting_')]
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    feature_columns = operational_settings + sensor_cols
    
    # Handle any remaining NaN or infinite values before scaling
    for col in feature_columns:
        # Replace any infinite values with NaN
        data[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN with median of the column
        data[col].fillna(data[col].median(), inplace=True)
    
    if is_train:
        scaler = MinMaxScaler()
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
    else:
        data[feature_columns] = scaler.transform(data[feature_columns])
    
    return data, scaler

def train_regression_model(X_train, y_train):
    """
    Train a regression model to predict Remaining Useful Life (RUL).
    """
    # Create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train a more robust RandomForestRegressor with hyperparameter tuning
    print("Training Random Forest Regressor with hyperparameter tuning...")
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Use a smaller subset for GridSearchCV to speed up training
    n_samples = min(5000, len(X_train_split))
    X_grid_search = X_train_split.iloc[:n_samples]
    y_grid_search = y_train_split.iloc[:n_samples]
    
    # Create and train the grid search
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_grid_search, y_grid_search)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Train model with best parameters on all training data
    regressor = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    regressor.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred_raw = regressor.predict(X_val)
    
    # Rescale predictions to the range of true RUL values
    y_min = y_train.min()
    y_max = y_train.max()
    print(f"True RUL range: {y_min:.4f} to {y_max:.4f}")
    
    # Create scaling function directly instead of using MinMaxScaler
    def scale_predictions(preds, y_min, y_max):
        # Get min and max of predictions
        pred_min = preds.min()
        pred_max = preds.max()
        
        # If prediction range is too small, use a constant shift
        if pred_max - pred_min < 1e-6:
            return np.ones_like(preds) * ((y_max + y_min) / 2)
        
        # Scale to [0, 1] then to [y_min, y_max]
        scaled = (preds - pred_min) / (pred_max - pred_min)
        scaled = scaled * (y_max - y_min) + y_min
        return scaled
    
    # Apply scaling
    y_val_pred = scale_predictions(y_val_pred_raw, y_min, y_max)
    
    print(f"Raw prediction range: {y_val_pred_raw.min():.4f} to {y_val_pred_raw.max():.4f}")
    print(f"Scaled prediction range: {y_val_pred.min():.4f} to {y_val_pred.max():.4f}")
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R² Score: {r2:.4f}")
    
    # Train on all training data for final model
    final_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    
    return final_model, (y_min, y_max)

def create_failure_labels(y, threshold=30):
    """
    Create binary labels for failure prediction.
    Engines with RUL <= threshold are marked as likely to fail (1),
    others as not likely to fail (0).
    """
    return (y <= threshold).astype(int)

def train_classification_model(X_train, failure_labels):
    """
    Train a classification model to predict failure likelihood.
    """
    # Create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, failure_labels, test_size=0.2, random_state=42, stratify=failure_labels
    )
    
    # Train RandomForestClassifier
    print("Training Random Forest Classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = classifier.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    
    # Check if there's more than one class in validation predictions
    unique_classes = np.unique(np.concatenate([y_val, y_val_pred]))
    if len(unique_classes) > 1:
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        
        # Create confusion matrix
        try:
            cm = confusion_matrix(y_val, y_val_pred)
            
            # Plot confusion matrix
            if plotting_available:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                plt.savefig('confusion_matrix.png')
        except Exception as e:
            print(f"Warning: Could not create confusion matrix: {e}")
    else:
        # Only one class
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Warning: Only one class present in validation data. Precision, recall, and F1 not calculated.")
    
    # Train on all training data for final model
    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, failure_labels)
    
    return final_model

def evaluate_models(model, clf_model, X_test, true_rul, threshold, test_data, rul_range=None):
    """Evaluate regression and classification models on the test data."""
    # Predict RUL
    y_pred_rul = model.predict(X_test)
    
    # Scale predictions if rul_range is provided
    if rul_range:
        raw_min, raw_max = y_pred_rul.min(), y_pred_rul.max()
        print(f"Raw prediction range: {raw_min:.4f} to {raw_max:.4f}")
        
        true_min, true_max = true_rul.min(), true_rul.max()
        
        if raw_min != true_min or raw_max != true_max:
            # Scale to match the range of true RUL
            y_pred_rul = np.clip(y_pred_rul, raw_min, raw_max)
            y_pred_rul = (y_pred_rul - raw_min) / (raw_max - raw_min) * (true_max - true_min) + true_min
            y_pred_rul = np.round(y_pred_rul)
            print(f"Scaled prediction range: {y_pred_rul.min():.4f} to {y_pred_rul.max():.4f}")
            
    print(f"True RUL range: {true_rul.min():.4f} to {true_rul.max():.4f}")
    
    # Predict failure probability
    if hasattr(clf_model, 'predict_proba'):
        y_pred_failure_prob = clf_model.predict_proba(X_test)[:, 1]
        y_pred_failure_clf = (y_pred_failure_prob >= 0.5).astype(int)
    else:
        y_pred_failure_clf = clf_model.predict(X_test)
    
    # Get unique engine IDs in test data
    unique_engines = test_data['engine_id'].unique()
    
    # Only keep the engines that have true RUL values
    n_engines_with_rul = len(true_rul)
    
    print(f"Number of engines in test data: {len(unique_engines)}")
    print(f"Number of engines with true RUL values: {n_engines_with_rul}")
    
    # Initialize arrays to store last prediction for each engine
    max_engines = min(len(unique_engines), n_engines_with_rul)
    unique_engines = unique_engines[:max_engines]
    true_rul_cut = true_rul[:max_engines]  # Limit true_rul to match available engines
    
    print(f"Using first {max_engines} engines for evaluation")
    
    last_rul_predictions = np.zeros(max_engines)
    last_failure_predictions = np.zeros(max_engines)
    
    # Get the last prediction for each engine
    for i, engine_id in enumerate(unique_engines):
        engine_mask = test_data['engine_id'] == engine_id
        engine_indices = np.where(engine_mask)[0]
        if len(engine_indices) > 0:  # Make sure the engine exists in test data
            last_idx = engine_indices[-1]  # Get the last index for this engine
            last_rul_predictions[i] = y_pred_rul[last_idx]
            last_failure_predictions[i] = y_pred_failure_clf[last_idx]
    
    # Convert RUL predictions to binary failure predictions
    last_rul_failure_predictions = (last_rul_predictions <= threshold).astype(int)
    
    # Calculate regression metrics
    mse = mean_squared_error(true_rul_cut, last_rul_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_rul_cut, last_rul_predictions)
    
    print("\nTest Results:")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Convert true RUL to binary failure labels
    true_failure = (true_rul_cut <= threshold).astype(int)
    
    # Check if there's more than one class in the predictions
    reg_unique_classes = np.unique(np.concatenate([true_failure, last_rul_failure_predictions]))
    clf_unique_classes = np.unique(np.concatenate([true_failure, last_failure_predictions]))
    
    # Calculate classification metrics for regression-based predictions
    reg_accuracy = accuracy_score(true_failure, last_rul_failure_predictions)
    
    if len(reg_unique_classes) > 1:
        reg_precision = precision_score(true_failure, last_rul_failure_predictions, zero_division=0)
        reg_recall = recall_score(true_failure, last_rul_failure_predictions, zero_division=0)
        reg_f1 = f1_score(true_failure, last_rul_failure_predictions, zero_division=0)
        
        print("\nFailure Classification Metrics (Regression-based):")
        print(f"Accuracy: {reg_accuracy:.4f}")
        print(f"Precision: {reg_precision:.4f}")
        print(f"Recall: {reg_recall:.4f}")
        print(f"F1 Score: {reg_f1:.4f}")
    else:
        print("\nFailure Classification Metrics (Regression-based):")
        print(f"Accuracy: {reg_accuracy:.4f}")
        print("Warning: Only one class present in regression-based predictions. Precision, recall, and F1 not calculated.")
    
    # Calculate classification metrics for classifier predictions
    clf_accuracy = accuracy_score(true_failure, last_failure_predictions)
    
    if len(clf_unique_classes) > 1:
        clf_precision = precision_score(true_failure, last_failure_predictions, zero_division=0)
        clf_recall = recall_score(true_failure, last_failure_predictions, zero_division=0)
        clf_f1 = f1_score(true_failure, last_failure_predictions, zero_division=0)
        
        print("\nFailure Classification Metrics (Classifier):")
        print(f"Accuracy: {clf_accuracy:.4f}")
        print(f"Precision: {clf_precision:.4f}")
        print(f"Recall: {clf_recall:.4f}")
        print(f"F1 Score: {clf_f1:.4f}")
        
        # Create confusion matrix for classifier
        try:
            cm = confusion_matrix(true_failure, last_failure_predictions)
            
            # Plot confusion matrix
            if plotting_available:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Classifier Confusion Matrix')
                plt.savefig('classifier_confusion_matrix.png')
        except Exception as e:
            print(f"Warning: Could not create confusion matrix: {e}")
    else:
        print("\nFailure Classification Metrics (Classifier):")
        print(f"Accuracy: {clf_accuracy:.4f}")
        print("Warning: Only one class present in classifier predictions. Precision, recall, and F1 not calculated.")
    
    return y_pred_rul, y_pred_failure_clf, last_rul_predictions

def prepare_data_for_lstm(X_train, y_train, X_test, sequence_length=10):
    """
    Prepare data for LSTM model by creating sequences of observations.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        sequence_length: Length of sequence for LSTM input
        
    Returns:
        X_train_seq: Sequence data for training
        y_train_seq: Target values for training
        X_test_seq: Sequence data for testing
        test_indices: Indices mapping from original test data to sequences
    """
    if not tensorflow_available:
        print("TensorFlow/Keras is not available. Skipping LSTM data preparation.")
        return None, None, None, None
    
    # Get engine_id column which we need for grouping but don't want in the features
    train_engine_ids = X_train['engine_id'].values
    test_engine_ids = X_test['engine_id'].values
    
    # Remove engine_id from features
    feature_cols = [col for col in X_train.columns if col != 'engine_id']
    X_train_features = X_train[feature_cols]
    X_test_features = X_test[feature_cols]
    
    # Convert to numpy arrays
    X_train_np = X_train_features.values
    y_train_np = y_train.values
    X_test_np = X_test_features.values
    
    # Get unique engine IDs
    unique_train_engines = np.unique(train_engine_ids)
    unique_test_engines = np.unique(test_engine_ids)
    
    X_train_seq = []
    y_train_seq = []
    X_test_seq = []
    test_indices = []
    
    # Process training data
    for engine_id in unique_train_engines:
        # Get indices for this engine
        engine_indices = np.where(train_engine_ids == engine_id)[0]
        engine_data = X_train_np[engine_indices]
        engine_target = y_train_np[engine_indices]
        
        # Skip if not enough data for a sequence
        if len(engine_data) < sequence_length:
            continue
        
        # Create sequences
        for i in range(len(engine_data) - sequence_length + 1):
            X_train_seq.append(engine_data[i:i+sequence_length])
            # Use the RUL at the end of the sequence as the target
            y_train_seq.append(engine_target[i+sequence_length-1])
    
    # Process test data
    for engine_id in unique_test_engines:
        # Get indices for this engine
        engine_indices = np.where(test_engine_ids == engine_id)[0]
        engine_data = X_test_np[engine_indices]
        
        # Skip if not enough data for a sequence
        if len(engine_data) < sequence_length:
            continue
        
        # Create sequences
        for i in range(len(engine_data) - sequence_length + 1):
            X_test_seq.append(engine_data[i:i+sequence_length])
            # Store the original index for mapping back to test data
            test_indices.append(engine_indices[i+sequence_length-1])
    
    # Convert to numpy arrays
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("Warning: No sequences could be created. Check your data.")
        return None, None, None, None
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    X_test_seq = np.array(X_test_seq)
    test_indices = np.array(test_indices)
    
    print(f"Created {len(X_train_seq)} training sequences and {len(X_test_seq)} test sequences")
    
    return X_train_seq, y_train_seq, X_test_seq, test_indices

def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Build an LSTM model for RUL prediction.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled LSTM model
    """
    if not tensorflow_available:
        print("TensorFlow/Keras is not available. Skipping LSTM model building.")
        return None
    
    # Create model
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    
    # Compile model with Adam optimizer and MSE loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    
    return model

def train_lstm_model(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train an LSTM model for RUL prediction.
    
    Args:
        X_train: Training data sequences
        y_train: Training target values
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
        
    Returns:
        Trained LSTM model
    """
    if not tensorflow_available:
        print("TensorFlow/Keras is not available. Skipping LSTM model training.")
        return None
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history if plotting is available
    if plotting_available:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('lstm_training_history.png')
    
    return model

def train_gradient_boosting_regressor(X_train, y_train):
    """
    Train a Gradient Boosting Regressor for RUL prediction.
    """
    # Create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    
    # Use a smaller subset for GridSearchCV to speed up training
    n_samples = min(5000, len(X_train_split))
    X_grid_search = X_train_split.iloc[:n_samples]
    y_grid_search = y_train_split.iloc[:n_samples]
    
    print("Training Gradient Boosting Regressor with hyperparameter tuning...")
    
    # Create and train the grid search
    grid_search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_grid_search, y_grid_search)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Train model with best parameters on all training data
    gbr = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    gbr.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = gbr.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R² Score: {r2:.4f}")
    
    # Train on all training data for final model
    final_model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    
    return final_model

def train_xgboost_regressor(X_train, y_train):
    """
    Train an XGBoost Regressor for RUL prediction.
    """
    if not xgboost_available:
        print("XGBoost not available. Skipping XGBoost regressor training.")
        return None
    
    # Create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3]
    }
    
    # Use a smaller subset for GridSearchCV to speed up training
    n_samples = min(5000, len(X_train_split))
    X_grid_search = X_train_split.iloc[:n_samples]
    y_grid_search = y_train_split.iloc[:n_samples]
    
    print("Training XGBoost Regressor with hyperparameter tuning...")
    
    # Create and train the grid search
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_grid_search, y_grid_search)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Train model with best parameters on validation set
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_child_weight=best_params['min_child_weight'],
        random_state=42
    )
    xgb_reg.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = xgb_reg.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R² Score: {r2:.4f}")
    
    # Train on all training data for final model
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_child_weight=best_params['min_child_weight'],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    
    return final_model

def train_xgboost_classifier(X_train, failure_labels):
    """
    Train an XGBoost Classifier for failure prediction.
    """
    if not xgboost_available:
        print("XGBoost not available. Skipping XGBoost classifier training.")
        return None
    
    # Create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, failure_labels, test_size=0.2, random_state=42, stratify=failure_labels
    )
    
    print("Training XGBoost Classifier...")
    
    # Train XGBoost classifier
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    clf.fit(X_train_split, y_train_split)
    
    # Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    
    # Check if there's more than one class in the predictions
    unique_classes = np.unique(np.concatenate([y_val, y_val_pred]))
    if len(unique_classes) > 1:
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        
        # Create confusion matrix
        try:
            cm = confusion_matrix(y_val, y_val_pred)
            
            # Plot confusion matrix
            if plotting_available:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('XGBoost Confusion Matrix')
                plt.savefig('xgboost_confusion_matrix.png')
        except Exception as e:
            print(f"Warning: Could not create confusion matrix: {e}")
    else:
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("Warning: Only one class present in validation data. Precision, recall, and F1 not calculated.")
    
    # Train on all training data for final model
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    final_model.fit(X_train, failure_labels)
    
    return final_model

# Load pre-processed data
def main():
    print("Loading processed data...")
    train_data = pd.read_csv('processed_train.csv')
    test_data = pd.read_csv('processed_test.csv')
    
    # Load true RUL values for test data
    true_rul = pd.read_csv('CMaps/RUL_FD001.txt', header=None)[0].values
    
    # Add more engineered features
    print("\nAdding engineered features...")
    train_data = add_engineered_features(train_data)
    test_data = add_engineered_features(test_data)
    
    # Split into features and target
    feature_cols = [col for col in train_data.columns if col not in ['engine_id', 'time_in_cycles', 'RUL', 'max_cycle', 'life_pct']]
    X_train = train_data[feature_cols]
    y_train = train_data['RUL']
    X_test = test_data[feature_cols]
    
    # Find an appropriate failure threshold based on data distribution
    q25 = np.percentile(true_rul, 25)
    print(f"\nTrue RUL distribution - 25th percentile: {q25:.1f}")
    
    # Using 25th percentile of true RUL as the threshold
    failure_threshold = q25  
    print(f"Setting failure threshold to {failure_threshold:.1f}")
    
    # Cap the RUL values at 125 (recommended in some NASA publications)
    # This can help focus the model on the degradation phase
    y_train_capped = y_train.copy()
    y_train_capped = y_train_capped.clip(upper=125)
    print(f"Capped RUL range: {y_train_capped.min():.1f} to {y_train_capped.max():.1f}")
    
    # Create binary failure labels (1 = likely to fail, 0 = not likely to fail)
    failure_labels = create_failure_labels(y_train, failure_threshold)
    
    # Check class distribution
    failure_count = failure_labels.sum()
    non_failure_count = len(failure_labels) - failure_count
    print(f"Class distribution - Failure: {failure_count} ({failure_count/len(failure_labels)*100:.1f}%), Non-failure: {non_failure_count} ({non_failure_count/len(failure_labels)*100:.1f}%)")
    
    # Feature selection to reduce dimensionality
    print("\nPerforming feature selection...")
    X_train_selected, X_test_selected = select_features(X_train, y_train_capped, X_test, n_features=30)
    
    # Store best models
    best_reg_model = None
    best_reg_rmse = float('inf')
    best_reg_name = None
    
    best_clf_model = None
    best_clf_f1 = 0
    best_clf_name = None
    
    # Train different regression models
    print("\n=== Training Regression Models ===")
    
    # Random Forest
    print("\nTraining Random Forest Regressor...")
    rf_model, rul_range = train_regression_model(X_train_selected, y_train_capped)
    
    # Gradient Boosting
    print("\nTraining Gradient Boosting Regressor...")
    gb_model = train_gradient_boosting_regressor(X_train_selected, y_train_capped)
    
    # XGBoost if available
    xgb_model = None
    if xgboost_available:
        print("\nTraining XGBoost Regressor...")
        xgb_model = train_xgboost_regressor(X_train_selected, y_train_capped)
    
    # Train LSTM model if TensorFlow is available
    lstm_model = None
    if tensorflow_available:
        print("\nPreparing data for LSTM...")
        # Clone the dataframes to avoid SettingWithCopyWarning
        X_train_lstm = X_train_selected.copy()
        X_train_lstm['engine_id'] = train_data['engine_id'].values
        
        X_test_lstm = X_test_selected.copy()
        X_test_lstm['engine_id'] = test_data['engine_id'].values
        
        print("Creating sequences for LSTM...")
        X_train_seq, y_train_seq, X_test_seq, test_indices = prepare_data_for_lstm(
            X_train_lstm, y_train_capped, X_test_lstm, sequence_length=10
        )
        
        if X_train_seq is not None:
            print("\nTraining LSTM model...")
            lstm_model = train_lstm_model(X_train_seq, y_train_seq, epochs=30, batch_size=32)
    
    # Train classification models
    print("\n=== Training Classification Models ===")
    
    # Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_clf_model = train_classification_model(X_train_selected, failure_labels)
    
    # XGBoost Classifier if available
    xgb_clf_model = None
    if xgboost_available:
        print("\nTraining XGBoost Classifier...")
        try:
            xgb_clf_model = train_xgboost_classifier(X_train_selected, failure_labels)
        except ValueError as e:
            print(f"Error training XGBoost classifier: {e}")
            print("Skipping XGBoost classifier...")
    
    # Evaluate models on test data
    print("\n=== Evaluating Models on Test Data ===")
    
    # Evaluate Random Forest
    print("\nEvaluating Random Forest Regressor...")
    rf_pred_rul, rf_pred_failure, rf_last_preds = evaluate_models(
        rf_model, rf_clf_model, X_test_selected, true_rul, failure_threshold, test_data, rul_range
    )
    
    # Make sure true_rul has the same length as rf_last_preds
    max_engines = min(len(test_data['engine_id'].unique()), len(true_rul))
    true_rul_cut = true_rul[:max_engines]
    
    # Calculate RF RMSE for comparison
    rf_rmse = np.sqrt(mean_squared_error(true_rul_cut, rf_last_preds))
    if rf_rmse < best_reg_rmse:
        best_reg_model = rf_model
        best_reg_rmse = rf_rmse
        best_reg_name = "Random Forest"
    
    # Get RF F1 score
    true_failure = (true_rul_cut <= failure_threshold).astype(int)
    rf_failure_preds = (rf_last_preds <= failure_threshold).astype(int)
    rf_f1 = f1_score(true_failure, rf_failure_preds, zero_division=0)
    if rf_f1 > best_clf_f1:
        best_clf_model = rf_clf_model
        best_clf_f1 = rf_f1
        best_clf_name = "Random Forest"
    
    # Evaluate Gradient Boosting
    print("\nEvaluating Gradient Boosting Regressor...")
    gb_pred_rul, gb_pred_failure, gb_last_preds = evaluate_models(
        gb_model, rf_clf_model, X_test_selected, true_rul, failure_threshold, test_data, rul_range
    )
    
    # Calculate GB RMSE for comparison
    gb_rmse = np.sqrt(mean_squared_error(true_rul_cut, gb_last_preds))
    if gb_rmse < best_reg_rmse:
        best_reg_model = gb_model
        best_reg_rmse = gb_rmse
        best_reg_name = "Gradient Boosting"
    
    # Evaluate XGBoost if available
    if xgboost_available and xgb_model is not None:
        print("\nEvaluating XGBoost Regressor...")
        xgb_pred_rul, xgb_pred_failure, xgb_last_preds = evaluate_models(
            xgb_model, xgb_clf_model if xgb_clf_model is not None else rf_clf_model, 
            X_test_selected, true_rul, failure_threshold, test_data, rul_range
        )
        
        # Calculate XGB RMSE for comparison
        xgb_rmse = np.sqrt(mean_squared_error(true_rul_cut, xgb_last_preds))
        if xgb_rmse < best_reg_rmse:
            best_reg_model = xgb_model
            best_reg_rmse = xgb_rmse
            best_reg_name = "XGBoost"
        
        # Get XGB F1 score if XGBoost classifier is available
        if xgb_clf_model is not None:
            xgb_failure_preds = (xgb_last_preds <= failure_threshold).astype(int)
            xgb_f1 = f1_score(true_failure, xgb_failure_preds, zero_division=0)
            if xgb_f1 > best_clf_f1:
                best_clf_model = xgb_clf_model
                best_clf_f1 = xgb_f1
                best_clf_name = "XGBoost"
    
    # Evaluate LSTM if available
    if tensorflow_available and lstm_model is not None:
        print("\nEvaluating LSTM...")
        try:
            # Predict on test sequences
            test_preds = lstm_model.predict(X_test_seq, verbose=0)
            
            # Calculate RMSE and MAE for last prediction of each engine
            # Get unique engine IDs from test data
            unique_engines = test_data['engine_id'].unique()
            last_lstm_preds = np.zeros(max_engines)
            
            for i, engine_id in enumerate(unique_engines[:max_engines]):
                # Get indices for this engine
                engine_indices = np.where(test_data['engine_id'].values == engine_id)[0]
                
                # Get all predictions for this engine from test_indices
                engine_test_indices = [idx for idx in range(len(test_indices)) if test_indices[idx] in engine_indices]
                
                if engine_test_indices:
                    # Get the last prediction
                    last_idx = engine_test_indices[-1]
                    last_lstm_preds[i] = test_preds[last_idx][0]
            
            # Calculate LSTM RMSE
            lstm_rmse = np.sqrt(mean_squared_error(true_rul_cut, last_lstm_preds))
            print(f"LSTM Test RMSE: {lstm_rmse:.4f}")
            
            if lstm_rmse < best_reg_rmse:
                best_reg_rmse = lstm_rmse
                best_reg_name = "LSTM"
            
            # Calculate LSTM MAE
            lstm_mae = mean_absolute_error(true_rul_cut, last_lstm_preds)
            print(f"LSTM Test MAE: {lstm_mae:.4f}")
            
            # Calculate failure metrics
            lstm_failure_preds = (last_lstm_preds <= failure_threshold).astype(int)
            lstm_f1 = f1_score(true_failure, lstm_failure_preds, zero_division=0)
            
            print(f"LSTM Failure F1 Score: {lstm_f1:.4f}")
            
            if lstm_f1 > best_clf_f1:
                best_clf_f1 = lstm_f1
                best_clf_name = "LSTM (thresholded)"
        except Exception as e:
            print(f"Error evaluating LSTM: {e}")
    
    # Print best models
    print("\n=== Best Models ===")
    print(f"Best Regression Model: {best_reg_name} (RMSE: {best_reg_rmse:.4f})")
    print(f"Best Classification Model: {best_clf_name} (F1: {best_clf_f1:.4f})")
    
    # Save best models
    if best_reg_name == "Random Forest":
        joblib.dump(rf_model, 'best_regression_model.pkl')
    elif best_reg_name == "Gradient Boosting":
        joblib.dump(gb_model, 'best_regression_model.pkl')
    elif best_reg_name == "XGBoost" and xgboost_available:
        joblib.dump(xgb_model, 'best_regression_model.pkl')
    # LSTM models are saved during training with a callback
    
    if best_clf_name == "Random Forest":
        joblib.dump(rf_clf_model, 'best_classification_model.pkl')
    elif best_clf_name == "XGBoost" and xgboost_available:
        joblib.dump(xgb_clf_model, 'best_classification_model.pkl')
    
    print("\nModels saved to 'best_regression_model.pkl' and 'best_classification_model.pkl'")
    
    # Feature importance for best model if it's a tree-based model
    if best_reg_name in ["Random Forest", "Gradient Boosting", "XGBoost"] and plotting_available:
        print("\nTop 15 important features:")
        
        if best_reg_name == "Random Forest":
            importances = rf_model.feature_importances_
        elif best_reg_name == "Gradient Boosting":
            importances = gb_model.feature_importances_
        elif best_reg_name == "XGBoost":
            importances = xgb_model.feature_importances_
        
        indices = np.argsort(importances)[-15:]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("Feature importance plot saved to 'feature_importance.png'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
