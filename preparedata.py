import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
DATA_PATH = os.path.join('raw_data') 
RANDOM_SEED = 42 # For reproducible results

# --- Step 1: Define Label Mapping ---
# 1. Iterate through your data directory to find all unique activity labels.
# 2. Assign an integer to each unique label found.

def get_label_mapping(data_path):
    """Generates a mapping of text labels to integers and vice versa."""
    all_files = os.listdir(data_path)
    unique_labels = sorted(list(set([f.split('_')[0] for f in all_files if f.endswith('.npy')])))
    
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    print("--- Activity Labels Found ---")
    print(label_to_int)
    print("-" * 30)
    return label_to_int, int_to_label

# --- Step 2: Load and Aggregate Data ---

def load_data(data_path, label_to_int):
    """Loads all sequences and creates the feature (X) and target (y) arrays."""
    X_data = [] # To store the sequences (features)
    y_labels = [] # To store the corresponding integer labels

    print("Loading data sequences...")

    for file_name in os.listdir(data_path):
        if file_name.endswith('.npy'):
            try:
                # Extract the text label (e.g., 'bicep_curl' from 'bicep_curl_1.npy')
                label_text = file_name.split('_')[0]
                label_int = label_to_int[label_text]
                
                # Load the sequence (60, 18) array
                sequence = np.load(os.path.join(data_path, file_name))
                
                # Check for correct shape (60 frames, 18 features) - Optional sanity check
                if sequence.shape == (60, 18):
                    X_data.append(sequence)
                    y_labels.append(label_int)
                else:
                    print(f"Skipping {file_name}: Expected shape (60, 18), got {sequence.shape}")

            except Exception as e:
                print(f"Error loading or processing file {file_name}: {e}")

    # Convert lists to NumPy arrays
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    
    print(f"\nTotal Sequences Loaded (X): {X.shape}")
    print(f"Total Labels Loaded (y): {y.shape}")
    return X, y

# --- Step 3: Split Data and One-Hot Encode ---

def prepare_for_model(X, y):
    """Splits data into train/validation/test sets and one-hot encodes labels."""
    
    # Do NOT one-hot encode y here yet. We use the integer labels (y) for splitting.
    
    # 1. Split into Training (80%) and Temporary (20%) using INTEGER labels for stratification
    # Note: We use y for stratification and get y_train_int and y_temp_int
    X_train, X_temp, y_train_int, y_temp_int = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # 2. Split Temporary (20%) into Validation (50% of temp) and Test (50% of temp)
    # Note: We use y_temp_int for stratification here
    X_val, X_test, y_val_int, y_test_int = train_test_split(
        X_temp, y_temp_int, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp_int
    )
    
    # 3. Now, one-hot encode the split integer labels for the Keras model
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_val = to_categorical(y_val_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)

    print("\n--- Final Dataset Shapes ---")
    print(f"Training Set (X_train): {X_train.shape}")
    print(f"Validation Set (X_val): {X_val.shape}")
    print(f"Testing Set (X_test): {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- Main Execution ---

if __name__ == '__main__':
    label_to_int, int_to_label = get_label_mapping(DATA_PATH)
    
    if not label_to_int:
        print("No .npy files found in the 'raw_data' directory. Cannot proceed.")
    else:
        # Load and aggregate data
        X, y = load_data(DATA_PATH, label_to_int)
        
        # Prepare for modeling
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_for_model(X, y)
        
        # OPTIONAL: Save the processed arrays for later use (highly recommended)
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)
        print("\nProcessed data saved as X_train.npy, y_train.npy, etc.")