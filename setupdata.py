import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration ---
SEQUENCE_LENGTH = 35  # Changed from 60 to 35 (~1 second at 30 FPS)
SAVED_JOINTS = [
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
]
DATA_PATH = os.path.join('raw_data')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"Created directory: {DATA_PATH}")

# --- Utility Functions ---

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

# --- Main Recorder Function ---

def run_recorder():
    """Initializes camera, captures video, extracts, normalizes, and saves data."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        sequence = []
        recording = False
        counting_down = False
        start_time = 0
        countdown_duration = 3  # Reduced to 2 seconds for shorter sequences
        sample_count = 0
        
        print("\nTARGET: Collect 50+ samples per class (100+ total)")
        print("\nControls:")
        print("   'S' - Start recording sequence")
        print("   'Q' - Quit application")
        print("="*70 + "\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            normalized_features = None
            if results.pose_landmarks:
                landmarks_list = results.pose_landmarks.landmark
                normalized_features = normalize_landmarks(landmarks_list)
            
            if counting_down:
                elapsed_time = time.time() - start_time
                remaining_time = countdown_duration - int(elapsed_time)
                
                if remaining_time > 0:
                    cv2.putText(image, f"GET READY: {remaining_time}", (150, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5, cv2.LINE_AA)
                    cv2.putText(image, "Start your motion NOW!", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
                else:
                    counting_down = False
                    recording = True
                    sequence = []
                    print("RECORDING!")
            
            elif recording:
                if normalized_features is not None:
                    sequence.append(normalized_features)
                    
                cv2.putText(image, f"CAPTURING: {len(sequence)}/{SEQUENCE_LENGTH} frames", 
                           (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4, cv2.LINE_AA)
                
                # Progress bar
                progress = len(sequence) / SEQUENCE_LENGTH
                bar_width = int(progress * 500)
                cv2.rectangle(image, (80, 250), (80 + bar_width, 280), (0, 255, 0), -1)
                cv2.rectangle(image, (80, 250), (580, 280), (255, 255, 255), 2)

                if len(sequence) >= SEQUENCE_LENGTH:
                    recording = False
                    sample_count += 1
                    sequence_array = np.array(sequence).astype(np.float32)
                    
                    print("\n" + "-"*70)
                    print(f"Sequence #{sample_count} captured!")
                    print("⌨Switch to terminal to enter label...")
                    current_label = input(f"Enter label (idle_1, lateralraise_1, etc.): ")
                    
                    file_name = f'{current_label}.npy'
                    save_path = os.path.join(DATA_PATH, file_name)
                    np.save(save_path, sequence_array)
                    
                    print(f"Saved: {save_path}")
                    print(f"Total samples: {sample_count}")
                    print("Press 'S' for next sample")
                    print("-"*70 + "\n")
                    
                    sequence = []
            else:
                cv2.putText(image, "Press 'S' to Start Recording", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Samples collected: {sample_count}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow('HAR Data Recorder - Press S to Record', image)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s') and not recording and not counting_down:
                counting_down = True
                start_time = time.time()
                print(f"\n⏱️  {countdown_duration}-second countdown starting...")

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print(f"🎬 Session complete! Collected {sample_count} samples")
    print(f"📁 Data saved to: {DATA_PATH}/")
    print("="*70 + "\n")

if __name__ == '__main__':
    run_recorder()