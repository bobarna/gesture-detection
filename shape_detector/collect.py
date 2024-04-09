# Add current directory to Python path 
import sys
import os
SCRIPT_DIR = os.path.abspath(os.pardir)
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


import cv2
from detector.detector import create_detector
from visu.visu import draw_landmarks_on_image
import mediapipe as mp
import numpy as np


import argparse
choices = os.listdir(DATA_DIR)
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shape", help="The shape you want to add data to", type=str, choices=choices, required=True)
parser.add_argument("-n", "--name", help="Your name", type=str, required=True)
parser.add_argument("-i", help="Number of data points you want to add (Default = 1000)", type=int, default=1000)

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

def main(args):
    # setup to write to the correct path
    write_path = os.path.join(DATA_DIR, args.shape, args.name)


    camera_capture = cv2.VideoCapture(0)

    if not camera_capture.isOpened():
        print("Cannot open camera")
        exit()

    ### Main Loop ###
    count = 1
    update = 50
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

        # hand shape detection
        pts = get_interest_points(detection_result)
        if pts != []:
            for hand in pts:
                with open(write_path, 'a') as file:
                    string = str(hand.flatten().tolist())[1:-2] + "\n"
                    file.write(string)
                count += 1

        # check if collected all points
        if count == args.i:
            break
        
        # print updates
        elif count > update:
            print(count)
            update = update + 50

        # draw on image
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
    main(parser.parse_args())