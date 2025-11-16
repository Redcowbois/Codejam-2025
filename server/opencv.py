import mediapipe as mp
import os
import numpy as np
import requests
import tensorflow as tf
import cv2
import time
import shared_state
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_LENGTH = 35
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'har_lstm_model', 'model.keras')
print(f"Checking Model Path: {MODEL_PATH}")
# Global vars
counter = 0
BROKER_HOST = "localhost"     
BROKER_PORT = 1883
MQTT_TOPIC_ENCOURAGEMENT = "encouragement"
LABEL_MAP = {0: 'IDLE', 1: 'LATERAL RAISE', 2: 'SQUAT'}


EXERCISE_CONFIG = {
    'LATERAL RAISE': {
        'JOINTS_FOR_CHECK': [
            {'hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, 'shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, 'elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, 'wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, 'side': 'Right'},
            {'hip': mp.solutions.pose.PoseLandmark.LEFT_HIP.value, 'shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, 'elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value, 'wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST.value, 'side': 'Left'},
        ],
        'THRESHOLDS': {
            'SHOULDER_MIN': 60,   # Must lift higher than this
            'SHOULDER_MAX': 120,  # Must lift lower than this (don't go too high)
            'ELBOW_MIN': 150,     # Must keep elbow straighter than this
        }
    },
    'SQUAT': {
        'JOINTS_FOR_CHECK': [
            # Placeholder: Would use Hip, Knee, Ankle for squat check
        ],
        'THRESHOLDS': {
            'KNEE_MIN': 80,  # Example: Must squat deeper than this knee angle
        }
    }
}


SAVED_JOINTS = [
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
]

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) between three points: A, B, and C.
    B is the vertex (the joint where the angle is formed).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 

    angle_deg = np.degrees(np.arccos(cosine_angle))

    return angle_deg




def check_form(landmarks, exercise_name):
    """
    Dynamically calculates key joint angles based on the exercise configuration.
    Input: The raw Mediapipe landmarks_list and the name of the exercise.
    Returns: A dictionary of form metrics including the final 'feedback'.
    """
    
    if exercise_name not in EXERCISE_CONFIG:
        return {'feedback': f"Error: Configuration not found for {exercise_name}"}

    config = EXERCISE_CONFIG[exercise_name]
    thresholds = config['THRESHOLDS']
    feedback = ""
    
    try:
        if exercise_name == 'LATERAL RAISE':
            
            for joint_set in config['JOINTS_FOR_CHECK']:
                side = joint_set['side']
                
                # 1. Get Coordinates of Key Joints
                shoulder_coords = [landmarks[joint_set['shoulder']].x, landmarks[joint_set['shoulder']].y, landmarks[joint_set['shoulder']].z]
                elbow_coords = [landmarks[joint_set['elbow']].x, landmarks[joint_set['elbow']].y, landmarks[joint_set['elbow']].z]
                wrist_coords = [landmarks[joint_set['wrist']].x, landmarks[joint_set['wrist']].y, landmarks[joint_set['wrist']].z]
                hip_coords = [landmarks[joint_set['hip']].x, landmarks[joint_set['hip']].y, landmarks[joint_set['hip']].z]
                
                # 2. Calculate Angles
                elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
                shoulder_angle = calculate_angle(hip_coords, shoulder_coords, elbow_coords)
                
                # 3. Form Check Logic
                if shoulder_angle < thresholds['SHOULDER_MIN']:
                    feedback += f"Lift {side} shoulder higher! "
                
                if shoulder_angle > thresholds['SHOULDER_MAX']:
                    feedback += f"Lower {side} shoulder slightly! "
                
                if elbow_angle < thresholds['ELBOW_MIN']:
                    feedback += f"Straighten {side} elbow! "
                
                # Attach angles to metrics for tracking (important for the LLM output)
                if side == 'Right':
                    form_metrics = {'r_shoulder_angle': shoulder_angle, 'r_elbow_angle': elbow_angle}
                else: # Left
                    form_metrics.update({'l_shoulder_angle': shoulder_angle, 'l_elbow_angle': elbow_angle})
                    
        # Add 'elif exercise_name == 'SQUAT':' here for future exercises
            
        if not feedback:
            feedback = "PERFECT FORM."
            
        form_metrics['feedback'] = feedback
        return form_metrics

    except Exception as e:
        # NOTE: This error now needs to be handled outside since form_metrics may not exist
        return {'feedback': f"Error: Cannot measure form ({e})"}




def normalize_landmarks(landmarks):
    """
    Translates and scales the joint coordinates to make the model invariant
    to person size or position on screen.
    """
    if not landmarks:
        # NOTE: This ensures the function returns a predictable structure even if no landmarks are detected.
        # It should match the size expected by your model (len(SAVED_JOINTS) * 3)
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



def opencv_run(socketio_backend):
    camera = cv2.VideoCapture(0)
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Stores the last SEQUENCE_LENGTH frames for prediction
    sequence = []
    
    # --- STABILITY VARIABLES ---
    LAST_PREDICTED_LABEL = 0           
    LOCK_TIMER_START = time.time()     
    LOCK_DURATION = 0.05                
    HIGH_CONFIDENCE_THRESHOLD = 0.75   
    
    # --- FEEDBACK LOCKOUT VARIABLES ---
    rep_in_progress = False           
    feedback_message = "Waiting for action..." 
    
    # --- REP PERFORMANCE TRACKING ---
    # We now track max/min for *both* left and right side to accommodate multiple exercises/limbs
    current_rep_max_r_shoulder = 0.0     
    current_rep_min_r_elbow = 180.0      
    current_rep_max_l_shoulder = 0.0
    current_rep_min_l_elbow = 180.0 
    
    # Variables that MUST be initialized outside the sequence block
    display_index = 0
    display_confidence = 1.0
    landmarks_list = None
    predicted_class_index = 0
    confidence = 1.0
    # ---------------------------------------------
    
    print("\n--- HAR Real-Time Predictor Initialized ---")
    
    last_prediction = "idle"
    while cap.isOpened():
            # ... (Frame capture, pose detection, feature extraction, sequence building remain the same) ...
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose Detection
        results = pose_detector.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame)
        shared_state.latest_jpeg = buffer.tobytes()
        if shared_state.flag == 1:
            # --- Feature Extraction and Sequence Building ---
            landmarks_list = None 
            if results.pose_landmarks:
                landmarks_list = results.pose_landmarks.landmark 
                normalized_features = normalize_landmarks(landmarks_list)
                
                sequence.append(normalized_features)
                
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                # 1. Raw Model Prediction
                input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                # res IS defined here
                res = model.predict(input_data, verbose=0)[0]
                
                predicted_class_index = np.argmax(res)
                confidence = res[predicted_class_index]
                
                # 2. Check for High-Confidence Action
                if predicted_class_index != 0 and confidence > HIGH_CONFIDENCE_THRESHOLD:
                    LAST_PREDICTED_LABEL = predicted_class_index
                    LOCK_TIMER_START = time.time()
                
                # 3. Apply Cooldown Lock (Aggressive IDLE transition)
                is_locked = time.time() - LOCK_TIMER_START < LOCK_DURATION
                is_idle_now = (predicted_class_index == 0)

                if is_locked and not is_idle_now:
                    display_index = LAST_PREDICTED_LABEL
                    display_confidence = confidence 
                else:
                    display_index = predicted_class_index
                    display_confidence = confidence
                        
                # Sequence shifting (to make response faster)
                sequence = sequence[15:]
            
            current_exercise_name = LABEL_MAP.get(display_index, 'IDLE')
            
            if display_index != 0 and landmarks_list is not None: 
                # Action is active or locked
                
                form_metrics = check_form(landmarks_list, current_exercise_name)
                
                if not rep_in_progress:
                    # --- START OF NEW REP: Reset metrics and calculate feedback ONCE ---
                    rep_in_progress = True
                    
                    # Reset performance metrics for the new rep
                    current_rep_max_r_shoulder = 0.0     
                    current_rep_min_r_elbow = 180.0      
                    current_rep_max_l_shoulder = 0.0
                    current_rep_min_l_elbow = 180.0 
                    
                    # Calculate and lock the initial feedback message
                    feedback_text = form_metrics['feedback']
                    feedback_message = f"{current_exercise_name.upper()}: {feedback_text} ({display_confidence*100:.1f}%)"
                
                # --- UPDATE WORST FORM METRICS (Only applicable to Lateral Raise for now) ---
                if current_exercise_name == 'LATERAL RAISE':
                    try:
                        # Right Side
                        if 'r_shoulder_angle' in form_metrics:
                            current_rep_max_r_shoulder = max(current_rep_max_r_shoulder, form_metrics['r_shoulder_angle'])
                            current_rep_min_r_elbow = min(current_rep_min_r_elbow, form_metrics['r_elbow_angle'])
                        # Left Side
                        if 'l_shoulder_angle' in form_metrics:
                            current_rep_max_l_shoulder = max(current_rep_max_l_shoulder, form_metrics['l_shoulder_angle'])
                            current_rep_min_l_elbow = min(current_rep_min_l_elbow, form_metrics['l_elbow_angle'])
                    except:
                        pass # Ignore if angle keys don't exist yet

                # DURING REP: Always display the LOCKED message
                current_prediction = feedback_message

            else: # display_index == 0 (IDLE)
                if rep_in_progress:
                    rep_in_progress = False
                    current_rep_max_l_shoulder = 0
                    current_rep_min_l_elbow = 180
                    current_rep_max_r_shoulder = 0
                    current_rep_min_r_elbow = 180
                    current_prediction = f"{current_exercise_name.upper()} ({display_confidence*100:.1f}%)"
                else:
                    current_prediction = f"{current_exercise_name.upper()} ({display_confidence*100:.1f}%)"
                feedback_message = "Waiting"

            current_split = current_prediction.split(" ")
            last_split = last_prediction.split(" ")
            if current_split[0] != last_split[0]:
                # print("MEOWW IN IF " + current_prediction)
                if (current_split[0] != "IDLE"):
                    index = current_prediction.find("(")
                    if shared_state.cur_exercise_flag == 1:
                        s = current_prediction.split(":")
                        s = s[0].lower()
                        s = s.title()

                        socketio_backend.emit("update_cur_exercise", s)
                        shared_state.cur_exercise_flag == 0

                    # If "(" exists, cut off everything from it (including)
                    if index != -1:
                        current_prediction = current_prediction[:index].strip()

                    socketio_backend.emit("type", current_prediction)  # broadcast to all clients
            
            last_prediction = current_prediction
            # print(current_prediction)

            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Display the prediction text 
            color = (0, 255, 0) if display_index == 1 else (0, 0, 255) 
            cv2.putText(image, current_prediction, (80, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5, cv2.LINE_AA)
        else:
            pass

    cap.release()
    cv2.destroyAllWindows()