#1. Capture Frames and Detect 2D Keypoints

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Video capture from file (or use cv2.VideoCapture(0) for webcam)
cap = cv2.VideoCapture(r'D:\UW_courses\Internship\JumpstartCSR\Videos\deadlift.mp4')

fps = 30  # frames per second
frame_count = 0

# Storage for keypoints
keypoints_2d = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure valid FPS division before performing the modulo operation
    frame_interval = max(int(cap.get(cv2.CAP_PROP_FPS) // fps), 1)  # Prevent division by zero

    # Only process frames at the specified interval
    if frame_count % frame_interval != 0:
        frame_count += 1
        continue

    # Process the frame (your logic here)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the keypoints
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        # Extract the 33 2D keypoints
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y])

        keypoints_2d.append(keypoints)

    # To display the frames, uncomment below (optional)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # Sample at 30 FPS
    # if frame_count % int(cap.get(cv2.CAP_PROP_FPS) // fps) != 0:
    #     continue
    
    frame_count += 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure valid FPS division before performing the modulo operation
    frame_interval = max(int(cap.get(cv2.CAP_PROP_FPS) // fps), 1)  # Prevent division by zero

    # Only process frames at the specified interval
    if frame_count % frame_interval != 0:
        frame_count += 1
        continue

    # Process the frame (your logic here)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Convert to a numpy array for further processing
keypoints_2d = np.array(keypoints_2d)

print(keypoints_2d.shape)

#2. Convert 2D Keypoints to 3D Keypoints
#VideoPose3D requires Detectron2, whose setup requires C compiler prerequisite
