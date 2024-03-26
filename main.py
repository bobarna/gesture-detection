# Add current directory to Python path 
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
from visu import draw_landmarks_on_image, get_coordinates
from detector.detector import create_detector
import mediapipe as mp
import numpy as np

camera_capture = cv2.VideoCapture(0)

if not camera_capture.isOpened():
    print("Cannot open camera")
    exit()

prev_coordinate_x = None
prev_coordinate_y = None

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

    print(detection_result)
    print(type(detection_result))

    #annotated_frame, current_coordinates_x, current_coordinates_y = draw_landmarks_on_image(
    annotated_frame = draw_landmarks_on_image(
        frame,
        detection_result
    )

    # First crack at velocity calculations for hand landmarks
    # Velocity currently represented as (change in distance) / frame
    current_coordinates_x, current_coordinates_y = get_coordinates()
    velocities = None
    if prev_coordinate_x == None and prev_coordinate_y == None:
        prev_coordinate_x = current_coordinates_x
        prev_coordinate_y = current_coordinates_y
    else:
        x1 = np.array(prev_coordinate_x)
        y1 = np.array(prev_coordinate_y)
        x2 = np.array(current_coordinates_x)
        y2 = np.array(current_coordinates_y)
        velocities = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        print(velocities)
        
    # Display the resulting frame
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Gesture Recognition', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
camera_capture.release()
cv2.destroyAllWindows()

