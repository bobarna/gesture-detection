# Based on https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
X_COORDINATES = None
Y_COORDINATES = None


def get_coordinates():
    return X_COORDINATES, Y_COORDINATES


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        # For the data format, see:
        # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Update global coordinates
        global X_COORDINATES
        X_COORDINATES = x_coordinates
        global Y_COORDINATES
        Y_COORDINATES = y_coordinates

        # Draw handedness (left or right hand) on the image.
        # if handedness[0].index == 0:
        #   category = "Left"
        # else:
        #   category = "Right"
        # cv2.putText(annotated_image, category,
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def crop_square(frame):
    # Determine the dimensions of the frame
    height, width = frame.shape[:2]

    # Find the size of the smallest dimension
    res = min(height, width)

    # Calculate cropping coordinates
    top = (height - res) // 2
    left = (width - res) // 2

    # Crop to a square frame
    frame = frame[top:top + res, left:left + res]

    # Resize to 512x512
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)

    return frame


PREV_MOUSE = None
CURR_COLOR = None


def get_mouse_data_from_hand_landmarks(hand_landmarks, res):
    mouse_data = np.zeros(8, dtype=np.float32)
    # Position information
    # swap coords for different coordinate system
    x_coord = np.mean(np.array([lm[1] for lm in hand_landmarks]))
    y_coord = np.mean(np.array([lm[0] for lm in hand_landmarks]))
    mxy = np.array([x_coord, y_coord], dtype=np.float32) * res
    #print(y_coord, x_coord, mxy, res)

    global PREV_MOUSE, CURR_COLOR
    if PREV_MOUSE is None:
        PREV_MOUSE = mxy
        CURR_COLOR = (np.random.rand(3) * 0.7) + 0.3
    else:
        mdir = mxy - PREV_MOUSE
        mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
        mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
        mouse_data[2], mouse_data[3] = mxy[0], mxy[1]

        mouse_data[4:7] = CURR_COLOR
        PREV_MOUSE = mxy

    return mouse_data


def reset_mouse():
    global PREV_MOUSE, CURR_COLOR
    PREV_MOUSE = None
    CURR_COLOR = None
