# Add current directory to Python path 
import sys
import os
import time

# Fluid simulator and landmarking imports
import sim.stable_fluid
import visu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
from visu.visu import draw_landmarks_on_image, crop_square
from detector.detector import create_detector
from shape_detector.shape_detect import classify
from bent_finger.shape import extend_or_bend
import mediapipe as mp
import numpy as np

import torch
import torch.nn as nn
from shape_detector.models.SimpleModel import SimpleModel

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    required=False,
    default="0",
    type=int,
    help="Camera index for machine. We found for Mac: 1, Windows: 0 (default: 0)",
)


def get_interest_points(detection_result):
    # extract
    hand_landmarks_list = detection_result.hand_landmarks
    pts = []
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        pts.append(np.array([([landmark.x, landmark.y, landmark.z]) for landmark in hand_landmarks]))

    return pts


RES = None
IS_OVERLAY_ON = False

def main(args):
    global IS_OVERLAY_ON

    camera_capture = cv2.VideoCapture(args.c)

    if not camera_capture.isOpened():
        print("Cannot open camera")
        exit()
    else:
        # Initialize fluid simulation
        ret, frame = camera_capture.read()
        frame = visu.visu.crop_square(frame)

        global RES
        RES = frame.shape[0]
        print(f"RES INITIALIZED AS {RES}")
        sim.stable_fluid.restart_simulation(RES)
        print(f"Fluid simulation initialized with resolution {RES}x{RES}")

    # Neural Network
    model = SimpleModel()
    model_name = "fist_model"
    path = os.path.join(SCRIPT_DIR, "shape_detector", "models", "saved_models", model_name)
    model.load_state_dict(torch.load(path))
    model.eval()

    # velocity variables
    prev_coordinate_x = None
    prev_coordinate_y = None

    # time tracking
    times = []

    ### Main Loop ###
    while True:
        ret, frame = camera_capture.read()
        frame = visu.visu.crop_square(frame)

        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Legacy code for velocity calculation. Removed for final submission

        #current_coordinates_x, current_coordinates_y = get_coordinates()
        #velocities = None
        #if prev_coordinate_x == None and prev_coordinate_y == None:
        #    prev_coordinate_x = current_coordinates_x
        #    prev_coordinate_y = current_coordinates_y
        #else:
        #    x1 = np.array(prev_coordinate_x)
        #    y1 = np.array(prev_coordinate_y)
        #    x2 = np.array(current_coordinates_x)
        #    y2 = np.array(current_coordinates_y)
        #    velocities = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # Process frame
        # mirror captured frame
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detector = create_detector()
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_frame)

        # bent finger detection
        left_eb = [''] * 5
        right_eb = [''] * 5
        num_hands = len(detection_result.handedness)
        if num_hands > 0:
            for i in range(num_hands):
                if (detection_result.handedness[i][0].index == 0):
                    left_eb = extend_or_bend(landmarks=detection_result.hand_landmarks[i])
                else:
                    right_eb = extend_or_bend(landmarks=detection_result.hand_landmarks[i])

        # Case where the only bent finger on either hand is the thumb
        only_thumb_bent = ['b', 'e', 'e', 'e', 'e']
        if left_eb == only_thumb_bent or right_eb == only_thumb_bent:
            sim.stable_fluid.GRAVITY_COEFF += 10.0
            if sim.stable_fluid.GRAVITY_COEFF >= 300:
                sim.stable_fluid.GRAVITY_COEFF = 300
        
        # Case where the only bent finger on either hand is the pinky
        only_pinky_bent = ['e', 'e', 'e', 'e', 'b']
        if left_eb == only_pinky_bent or right_eb == only_pinky_bent:
            sim.stable_fluid.GRAVITY_COEFF -= 10.0
            if sim.stable_fluid.GRAVITY_COEFF <= -300:
                sim.stable_fluid.GRAVITY_COEFF = -300

        # hand shape detection
        pts = get_interest_points(detection_result)
        mouse_data = np.zeros(8, dtype=np.float32)
        if pts != []:
            for hand in pts:
                st = time.time_ns() / 1000000
                pred = model(torch.tensor(hand.flatten(), dtype=torch.float32))

                label = classify(pred)
                ft = time.time_ns() / 1000000
                inference_time = ft - st


                # Display the resulting frame
                height, width, _ = frame.shape
                x_coordinates = [landmark[0] for landmark in hand]
                y_coordinates = [landmark[1] for landmark in hand]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height)

                # Draw handedness (left or right hand) on the image.

                cv2.putText(frame, label,
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            1, (0, 0, 0), 3, cv2.LINE_AA)

                if label == 'fist':
                    # get mouse data for stepping fluid simulation
                    mouse_data = visu.visu.get_mouse_data_from_hand_landmarks(hand, RES)
                else:
                    # generate new color
                    visu.visu.reset_mouse()

        sim.stable_fluid.step(mouse_data)

        if IS_OVERLAY_ON:
            # optionally, add camera frame as background
            frame = 0.3 * np.array((frame / 255.0), dtype=np.float32) + sim.stable_fluid.dyes_pair.cur.to_numpy()
        else:
            # Set fluid sim as background
            frame = sim.stable_fluid.dyes_pair.cur.to_numpy()
        
        
        cv2.putText(frame, f"Gravity: {sim.stable_fluid.GRAVITY_COEFF:.0f}",
                    (20, 500), cv2.FONT_HERSHEY_DUPLEX,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # draw landmarks on image
        annotated_frame = draw_landmarks_on_image(
            frame,
            detection_result
        )

        # Display the resulting frame
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Gesture Recognition', annotated_frame)

        # Press 'o' to toggle camera image overlay on/off
        if cv2.waitKey(1) & 0xFF == ord('o'):
            IS_OVERLAY_ON = not IS_OVERLAY_ON

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    camera_capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(parser.parse_args())
