import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time

# --- Configuration (Must Match Training Config) ---
# Location where your trained model was saved
MODEL_PATH = os.path.join('har_lstm_model', 'model.keras')
# The duration of one sample sequence (60 frames)
SEQUENCE_LENGTH = 35

# Define the SAVED_JOINTS list (must be identical to prepare_data.py)
SAVED_JOINTS = [
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
]

# Load the label mapping (assuming you know your classes, e.g., idle=0, lateralraise=1)
# NOTE: If you had more classes, you'd need to load this from a saved file.
# For now, manually define based on your data:
LABEL_MAP = {0: 'IDLE', 1: 'LATERAL RAISE'}


# --- Utility Function (Must be IDENTICAL to the one used for data capture) ---

def normalize_landmarks(landmarks):
    """
    Translates and scales the joint coordinates to make the model invariant
    to person size or position on screen.
    """
    if not landmarks:
        return np.zeros(len(SAVED_JOINTS) * 3)

    r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    r_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]

    center_x = (r_hip.x + l_hip.x) / 2
    center_y = (r_hip.y + l_hip.y) / 2
    center_z = (r_hip.z + l_hip.z) / 2
    center = np.array([center_x, center_y, center_z])

    scale_factor = np.sqrt(
        (r_shoulder.x - l_shoulder.x)**2 +
        (r_shoulder.y - l_shoulder.y)**2 +
        (r_shoulder.z - l_shoulder.z)**2
    )
    if scale_factor < 0.01:
        scale_factor = 1.0

    processed_features = []
    for joint_enum in SAVED_JOINTS:
        idx = joint_enum.value
        p = landmarks[idx]
        raw_coords = np.array([p.x, p.y, p.z])
        
        normalized_coords = (raw_coords - center) / scale_factor
        processed_features.extend(normalized_coords.tolist())

    return np.array(processed_features)


# --- Main Predictor Function ---

def run_predictor():
    """Loads model and runs real-time prediction loop."""
    
    # Load the trained Keras model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Stores the last SEQUENCE_LENGTH frames for prediction
    sequence = []
    current_prediction = "Waiting for sequence..."
    
    print("\n--- HAR Real-Time Predictor Initialized ---")
    print("Move to start filling the prediction sequence.")
    print("Press 'Q' to QUIT.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose Detection
        results = pose_detector.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Feature Extraction and Sequence Building ---
        if results.pose_landmarks:
            landmarks_list = results.pose_landmarks.landmark
            normalized_features = normalize_landmarks(landmarks_list)
            
            # Add new frame to the sequence
            sequence.append(normalized_features)
            
            # Keep the sequence length capped
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:] # Keep only the last 60 frames

        # --- Prediction Logic ---
        if len(sequence) == SEQUENCE_LENGTH:
            # Convert the list of 30 frames into the required model input shape (1, 35, 18)
            input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
            
            # Make the prediction
            res = model.predict(input_data, verbose=0)[0]
            
            # Get the index of the highest probability
            predicted_class_index = np.argmax(res)
            
            # Get the confidence (probability)
            confidence = res[predicted_class_index]
            
            # Update the display string
            current_prediction = f"{LABEL_MAP[predicted_class_index].upper()} ({confidence*100:.1f}%)"
            
            # OPTIONAL: Reset sequence periodically to avoid "laggy" updates
            # sequence = sequence[30:] # Uncomment to shift the window by 30 frames
            
        # --- Display Results ---
        # Draw MediaPipe Landmarks (Optional, but useful for visualization)
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Display the prediction text
        color = (0, 255, 0) if "IDLE" in current_prediction or "RAISE" in current_prediction else (0, 0, 255)
        cv2.putText(image, current_prediction, (100, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, color, 3, cv2.LINE_AA)

        cv2.imshow('HAR Predictor', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_predictor()