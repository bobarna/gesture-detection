# Add current directory to Python path 
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
from visu.visu import draw_landmarks_on_image
from detector.detector import create_detector
from bent_finger.shape import extend_or_bend
import mediapipe as mp
import numpy as np

import torch
import torch.nn as nn
from shape_detector.models.SimpleModel import SimpleModel

def get_interest_points(detection_result):
    # extract
    hand_landmarks_list = detection_result.hand_landmarks
    pts = []
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        pts = np.array([([landmark.x, landmark.y, landmark.z]) for landmark in hand_landmarks])

    return pts

def main():
    camera_capture = cv2.VideoCapture(0)

    if not camera_capture.isOpened():
        print("Cannot open camera")
        exit()

    # Neural Network
    model = SimpleModel()
    model_name = "simple_model"
    path = os.path.join(SCRIPT_DIR, "shape_detector", "models", "saved_models", model_name)
    model.load_state_dict(torch.load(path))
    model.eval()

    ### Main Loop ###
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

        # breakpoint()

        # bent finger detection
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

        # hand shape detection
        pts = get_interest_points(detection_result)
        if pts != []:
            pred = model(torch.tensor(pts.flatten(), dtype=torch.float32))
            # if pred[1] > pred[0]:
            #     print("Pointing Right")
            # else:
            #     print("Other")

            # Display the resulting frame
            height, width, _ = frame.shape
            x_coordinates = [landmark[0] for landmark in pts]
            y_coordinates = [landmark[1] for landmark in pts]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height)

            # Draw handedness (left or right hand) on the image.
            
            cv2.putText(frame, "pointing right" if pred[1]>pred[0] else "other",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 0, 0), 3, cv2.LINE_AA)





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


if __name__ == "__main__":
    main()
