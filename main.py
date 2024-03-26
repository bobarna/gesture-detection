# Add current directory to Python path 
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
from visu import draw_landmarks_on_image
from detector.detector import create_detector
from shape import extend_or_bend
import mediapipe as mp

camera_capture = cv2.VideoCapture(0)

if not camera_capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = camera_capture.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Process frame
    # mirror captured frame
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = create_detector()
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_frame)

    left_eb = [''] * 5
    right_eb = [''] * 5
    num_hands = len(detection_result.handedness)
    if num_hands > 0:
        # print(f"handedness: {detection_result.handedness}")
        # print(f"landmark: {detection_result.hand_landmarks[0][4].z}")
        for i in range(num_hands):
            if(detection_result.handedness[i][0].index == 0):
                left_eb = extend_or_bend(landmarks=detection_result.hand_landmarks[i])
                print(f"left hand shape: {left_eb}\n")
            else:
                right_eb = extend_or_bend(landmarks=detection_result.hand_landmarks[i])
                print(f"right hand shape: {right_eb}\n")

    annotated_frame = draw_landmarks_on_image(
        frame,
        detection_result
    )

    # Display the resulting frame
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Gesture Recognition', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
camera_capture.release()
cv2.destroyAllWindows()

