import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(train_file, test_file, rul_file=None):
    """
    Load and process the C-MAPSS dataset.
    """
    column_names = ['engine_id', 'time_in_cycles', 'op_setting_1',
                    'op_setting_2', 'op_setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
                   
    # Load train data
    train_data = pd.read_csv(train_file, sep=' ', header=None)
    # Drop the last two columns which contain NaN values
    train_data = train_data.iloc[:, :-2]
    train_data.columns = column_names
    
    # Load test data
    test_data = pd.read_csv(test_file, sep=' ', header=None)
    # Drop the last two columns which contain NaN values
    test_data = test_data.iloc[:, :-2]
    test_data.columns = column_names
    
    # Load RUL data if provided
    if rul_file:
        rul_data = pd.read_csv(rul_file, header=None)
        # Map RUL to test engines
        rul_dict = {}
        for idx, rul in enumerate(rul_data[0]):
            rul_dict[idx+1] = rul  # Engine IDs are 1-indexed
        
        # Convert engine_id to 1-indexed to match RUL file
        test_data['engine_id'] = test_data['engine_id'] + 1
    
    # Compute RUL for training data
    # Group by engine_id and get max cycle for each engine
    max_cycles = train_data.groupby('engine_id')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge with original data
    train_data = train_data.merge(max_cycles, on='engine_id')
    
    # Calculate RUL - number of cycles left to failure
    train_data['RUL'] = train_data['max_cycle'] - train_data['time_in_cycles']
    
    return train_data, test_data, rul_dict if rul_file else None

def select_sensors(train_data, test_data):
    """
    Select only the useful sensors based on variance and correlation.
    """
    # Drop constant sensors
    sensor_cols = [col for col in train_data.columns if col.startswith('sensor')]
    
    # Check variance in training data
    sensor_var = train_data[sensor_cols].var()
    constant_sensors = sensor_var[sensor_var < 0.001].index.tolist()
    
    print(f"Dropping {len(constant_sensors)} constant sensors: {constant_sensors}")
    
    # Drop sensors with very low variance
    selected_sensors = [col for col in sensor_cols if col not in constant_sensors]
    
    # Ensure these columns exist in both train and test
    # First get columns common to both datasets
    train_selected_cols = ['engine_id', 'time_in_cycles'] + selected_sensors
    test_selected_cols = ['engine_id', 'time_in_cycles'] + selected_sensors
    
    # Add RUL and max_cycle if they exist in each dataset
    if 'RUL' in train_data.columns:
        train_selected_cols.append('RUL')
    if 'max_cycle' in train_data.columns:
        train_selected_cols.append('max_cycle')
    
    train_data_selected = train_data[train_selected_cols]
    test_data_selected = test_data[test_selected_cols]
    
    return train_data_selected, test_data_selected

def add_remaining_operating_cycles(test_data, rul_dict):
    """
    Add RUL to test data based on the last cycle of each engine.
    """
    test_data_with_rul = test_data.copy()
    
    # Group by engine_id and get the max cycle
    engine_max_cycles = test_data.groupby('engine_id')['time_in_cycles'].max().reset_index()
    engine_max_cycles.columns = ['engine_id', 'max_test_cycle']
    
    # Add max_test_cycle to test_data
    test_data_with_rul = test_data_with_rul.merge(engine_max_cycles, on='engine_id')
    
    # Calculate RUL for each row in test data
    # First add the RUL at the end of the test sequence
    engine_rul = pd.DataFrame(list(rul_dict.items()), columns=['engine_id', 'end_rul'])
    test_data_with_rul = test_data_with_rul.merge(engine_rul, on='engine_id')
    
    # Then calculate RUL for each row: end_rul + (max_test_cycle - time_in_cycles)
    test_data_with_rul['RUL'] = test_data_with_rul['end_rul'] + (test_data_with_rul['max_test_cycle'] - test_data_with_rul['time_in_cycles'])
    
    # Drop temporary columns
    test_data_with_rul = test_data_with_rul.drop(['max_test_cycle', 'end_rul'], axis=1)
    
    return test_data_with_rul

def scale_features(train_data, test_data):
    """
    Scale all feature columns using MinMaxScaler.
    """
    # Get the columns that exist in both datasets
    common_feature_cols = [col for col in train_data.columns if col not in ['RUL', 'engine_id', 'max_cycle', 'life_pct']]
    common_feature_cols = [col for col in common_feature_cols if col in test_data.columns]
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Fit on training data
    scaler.fit(train_data[common_feature_cols])
    
    # Transform both datasets
    train_data_scaled = train_data.copy()
    test_data_scaled = test_data.copy()
    
    train_data_scaled[common_feature_cols] = scaler.transform(train_data[common_feature_cols])
    test_data_scaled[common_feature_cols] = scaler.transform(test_data[common_feature_cols])
    
    # Scale additional columns in train_data that are not in test_data
    train_only_cols = [col for col in train_data.columns if col not in ['RUL', 'engine_id'] 
                      and col not in common_feature_cols
                      and col not in ['max_cycle', 'life_pct']]
    
    if train_only_cols:
        train_scaler = MinMaxScaler()
        train_data_scaled[train_only_cols] = train_scaler.fit_transform(train_data[train_only_cols])
    
    # Handle max_cycle separately if it exists in train_data
    if 'max_cycle' in train_data.columns:
        max_cycle_scaler = MinMaxScaler()
        train_data_scaled['max_cycle'] = max_cycle_scaler.fit_transform(train_data[['max_cycle']])
    
    # Handle life_pct separately if it exists in train_data
    if 'life_pct' in train_data.columns:
        # life_pct is already in [0, 1] range, no need to scale
        pass
    
    return train_data_scaled, test_data_scaled

def add_engineered_features(data):
    """
    Add engineered features based on sensor readings.
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Dynamically get sensor columns that exist in the data
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    
    # Add rolling mean for each sensor (window size = 5)
    for col in sensor_cols:
        rolling_mean = data.groupby('engine_id')[col].rolling(
            window=5, min_periods=1).mean().reset_index(0, drop=True)
        data[f'{col}_rolling_mean'] = rolling_mean
    
    # Add rolling standard deviation (window size = 5)
    for col in sensor_cols:
        rolling_std = data.groupby('engine_id')[col].rolling(
            window=5, min_periods=1).std().reset_index(0, drop=True)
        # Fill NaN with 0 (for the first few readings with insufficient window)
        data[f'{col}_rolling_std'] = rolling_std.fillna(0)
    
    # Add rate of change for each sensor
    for col in sensor_cols:
        # Calculate rate of change
        rate = data.groupby('engine_id')[col].diff() / data.groupby('engine_id')['time_in_cycles'].diff()
        # Replace infinite values with NaN
        rate = rate.replace([np.inf, -np.inf], np.nan)
        # Fill NaN using forward fill and backward fill
        rate = rate.groupby(data['engine_id']).ffill().bfill()
        # If any NaN remain, fill with 0
        data[f'{col}_rate'] = rate.fillna(0)
    
    # Add time from start as percentage of total cycles (life percentage) if max_cycle exists
    if 'max_cycle' in data.columns:
        data['life_pct'] = data['time_in_cycles'] / data['max_cycle']
    
    return data

def preprocess_data():
    """Main preprocessing function."""
    print("Loading data...")
    train_data, test_data, rul_dict = load_data(
        'CMaps/train_FD001.txt', 
        'CMaps/test_FD001.txt', 
        'CMaps/RUL_FD001.txt'
    )
    
    print("Selecting important sensors...")
    train_data, test_data = select_sensors(train_data, test_data)
    
    print("Adding RUL to test data...")
    test_data = add_remaining_operating_cycles(test_data, rul_dict)
    
    print("Adding engineered features...")
    train_data = add_engineered_features(train_data)
    test_data = add_engineered_features(test_data)
    
    print("Scaling features...")
    train_data, test_data = scale_features(train_data, test_data)
    
    print("Saving processed data...")
    train_data.to_csv('processed_train.csv', index=False)
    test_data.to_csv('processed_test.csv', index=False)
    
    print("Preprocessing complete!")
    
    # Print summary statistics
    print("\nTraining data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    
    print("\nRUL statistics in training data:")
    print(train_data['RUL'].describe())
    
    print("\nRUL statistics in test data:")
    print(test_data['RUL'].describe())
    
    # Print class distribution based on RUL threshold of 30
    threshold = 30
    train_failure = (train_data['RUL'] <= threshold).sum()
    test_failure = (test_data['RUL'] <= threshold).sum()
    
    print(f"\nClass distribution with threshold {threshold}:")
    print(f"Training data - Failure: {train_failure} ({train_failure/len(train_data)*100:.1f}%), Non-failure: {len(train_data)-train_failure} ({(len(train_data)-train_failure)/len(train_data)*100:.1f}%)")
    print(f"Test data - Failure: {test_failure} ({test_failure/len(test_data)*100:.1f}%), Non-failure: {len(test_data)-test_failure} ({(len(test_data)-test_failure)/len(test_data)*100:.1f}%)")

if __name__ == "__main__":
    preprocess_data() 