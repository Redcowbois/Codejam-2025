import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration ---
# The duration of one sample sequence (60 frames at ~30 FPS is 2 seconds)
SEQUENCE_LENGTH = 60
# Features: Right/Left Shoulder, Elbow, Wrist (3 joints * 2 sides * 3 coords = 18)
# We will save the X, Y, Z coordinates for each of these 6 joints.
SAVED_JOINTS = [
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
]
DATA_PATH = os.path.join('raw_data') # Directory to save the sequences
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"Created directory: {DATA_PATH}")

# --- Utility Functions ---

def normalize_landmarks(landmarks):
    """
    Translates and scales the joint coordinates to make the model invariant
    to person size or position on screen.
    
    1. Centering (Translation): Use the average hip position as the origin (0,0,0).
    2. Scaling: Use the distance between the two shoulders as the reference unit.
    
    Args:
        landmarks: MediaPipe normalized landmarks object.
    
    Returns:
        numpy.ndarray: An array of shape (18,) containing normalized [x, y, z] for the 6 joints.
    """
    if not landmarks:
        # Return a zero array if landmarks are not detected (handles errors gracefully)
        return np.zeros(len(SAVED_JOINTS) * 3)

    # 1. Define Centering/Scaling Points
    r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    r_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]

    # Calculate the center point (Mid-Hip)
    center_x = (r_hip.x + l_hip.x) / 2
    center_y = (r_hip.y + l_hip.y) / 2
    center_z = (r_hip.z + l_hip.z) / 2
    center = np.array([center_x, center_y, center_z])

    # Calculate scaling factor (distance between shoulders)
    scale_factor = np.sqrt(
        (r_shoulder.x - l_shoulder.x)**2 +
        (r_shoulder.y - l_shoulder.y)**2 +
        (r_shoulder.z - l_shoulder.z)**2
    )
    # Prevent division by zero if the scale factor is tiny
    if scale_factor < 0.01:
        scale_factor = 1.0

    processed_features = []
    for joint_enum in SAVED_JOINTS:
        # Get the current joint's raw coordinates
        idx = joint_enum.value
        p = landmarks[idx]
        raw_coords = np.array([p.x, p.y, p.z])
        
        # Apply Translation and Scaling
        normalized_coords = (raw_coords - center) / scale_factor
        
        # Append the normalized (x, y, z) to the feature list
        processed_features.extend(normalized_coords.tolist())

    return np.array(processed_features)

# --- Main Recorder Function ---

def run_recorder():
    """Initializes camera, captures video, extracts, normalizes, and saves data."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_pose = mp.solutions.pose
    # Use 'static_image_mode=False' for video input
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        sequence = [] # Stores the 60 frames for the current sample
        recording = False
        counting_down = False # New state for the timer
        start_time = 0
        countdown_duration = 3 # 3 seconds pre-roll
        sample_count = 0
        current_label = ""
        
        print("\n--- Data Recorder Initialized ---")
        print("1. Press 'S' to START recording a sequence.")
        print("2. When recording, the status will show 'CAPTURING...'")
        print("3. After 60 frames, you will be prompted to enter a label.")
        print("4. Press 'Q' to QUIT the application.\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a more natural mirror view
            frame = cv2.flip(frame, 1)
            
            # Recolor image to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR for OpenCV display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # --- Draw Landmarks (before status text) ---
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # --- Extract and Process Features ---
            normalized_features = None
            if results.pose_landmarks:
                landmarks_list = results.pose_landmarks.landmark
                normalized_features = normalize_landmarks(landmarks_list)
            
            # --- Handle Recording and Countdown States ---
            if counting_down:
                elapsed_time = time.time() - start_time
                remaining_time = countdown_duration - int(elapsed_time)
                
                if remaining_time > 0:
                    # Display countdown timer
                    cv2.putText(image, f"GET READY: {remaining_time}", (200, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.putText(image, "Recording starts soon...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA) # Orange text
                else:
                    # Countdown finished, start recording
                    counting_down = False
                    recording = True
                    sequence = [] # Ensure sequence is clear on start
                    print("Recording started!")
            
            elif recording:
                if normalized_features is not None:
                    sequence.append(normalized_features)
                    
                # Update status display
                cv2.putText(image, f"CAPTURING: {len(sequence)}/{SEQUENCE_LENGTH} frames", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Check if sequence is complete
                if len(sequence) >= SEQUENCE_LENGTH:
                    recording = False
                    sample_count += 1
                    print("\n--- Sequence Captured ---")
                    
                    # Save the sequence
                    sequence_array = np.array(sequence).astype(np.float32)
                    
                    # Prompt user for label
                    # To ensure the input prompt appears correctly when running in a console environment,
                    # we must perform the blocking input operation *outside* the main OpenCV loop's thread.
                    # A common simple pattern for this type of script is to pause the video feed
                    # or prompt in the console while the video window is momentarily less reactive.
                    print("!!! Switch to the console/terminal to enter the label !!!")
                    current_label = input(f"Enter label for sample {sample_count} (e.g., bicep_curl_1): ")
                    
                    file_name = f'{current_label}.npy'
                    save_path = os.path.join(DATA_PATH, file_name)
                    
                    # Save the 60x18 array
                    np.save(save_path, sequence_array)
                    
                    print(f"Successfully saved sequence to {save_path}")
                    print("Press 'S' to record next sample.")
                    
                    # Reset sequence for the next capture (already done before save, but here for clarity)
                    sequence = []
            else:
                # Default status when waiting
                cv2.putText(image, "Press 'S' to Start Recording", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('HAR Data Recorder - Press S to Record', image)

            # Handle key presses
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                break
            
            # Start recording when 's' is pressed and not currently recording or counting down
            elif key & 0xFF == ord('s') and not recording and not counting_down:
                counting_down = True
                start_time = time.time()
                print(f"Starting {countdown_duration}-second countdown...")
                

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_recorder()