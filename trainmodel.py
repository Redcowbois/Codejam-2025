import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os

# --- Configuration ---
# Load your prepared data (assuming you ran the 'prepare_data.py' script)
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Get shape information from the loaded data
SEQUENCE_LENGTH = X_train.shape[1]  # Should be 60
FEATURE_DIMENSION = X_train.shape[2] # Should be 18 (6 joints * 3 coords)
NUM_CLASSES = y_train.shape[1] # Number of distinct activities (e.g., 3, 5, etc.)

# Training parameters
LOG_DIR = os.path.join('Logs')
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_NAME = 'har_lstm_model'
EPOCHS = 200 # You might need more or fewer depending on your data
BATCH_SIZE = 64

# --- Model Definition ---

def build_lstm_model():
    """Defines and compiles a Bidirectional LSTM model."""
    
    # Use Bidirectional LSTM to capture temporal dependencies in both forward and backward directions
    model = Sequential()
    
    # Layer 1: Bidirectional LSTM with 64 units
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu'), 
                            input_shape=(SEQUENCE_LENGTH, FEATURE_DIMENSION)))
    model.add(Dropout(0.2))

    # Layer 2: Standard LSTM with 128 units (return_sequences=False because it's the last sequential layer)
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    
    # Layer 3: Dense layer for processing LSTM output
    model.add(Dense(64, activation='relu'))
    
    # Output Layer: Softmax activation for multi-class classification
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    
    print("\n--- Model Summary ---")
    model.summary()
    
    return model

# --- Main Training Function ---

def train_model():
    """Builds the model, sets up callbacks, and starts training."""
    
    # 1. Build Model
    model = build_lstm_model()
    
    # 2. Define Callbacks
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    
    # Stop training early if validation loss doesn't improve for 10 epochs
    es_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    model_dir = MODEL_NAME # 'har_lstm_model'
    model_save_path = os.path.join(model_dir, 'model.keras')
    
    # >>> ADD THESE TWO LINES TO CREATE THE DIRECTORY <<<
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # >>> ------------------------------------------ <<<

    model.save(model_save_path)
    print(f"\n✅ Model training complete. Saved model to: {model_save_path}")
    
    # 3. Train Model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[tb_callback, es_callback]
    )
    
    # 4. Save the trained model
    model_save_path = os.path.join(MODEL_NAME, 'model.keras')
    model.save(model_save_path)
    print(f"\n✅ Model training complete. Saved model to: {model_save_path}")
    
    # Display the final validation accuracy
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    
    return model

if __name__ == '__main__':
    # Ensure TensorFlow uses the GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Only allocate the memory it needs
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU available and configured for training.")
        except RuntimeError as e:
            print(e)
            
    train_model()