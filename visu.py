# Based on https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import math

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  prev_coordinate_x = None
  prev_coordinate_y = None

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

    # First draft of velocity calculations
    # Velocity currently represented as change in distance / frame
    velocities = np.zeros(x_coordinates.shape[0])
    for coord_index in np.arange(x_coordinates.shape[0]):
        if prev_coordinate_x == None and prev_coordinate_y == None:
           velocities[coord_index] == 0
           prev_coordinate_x = x_coordinates[coord_index]
           prev_coordinate_y = y_coordinates[coord_index]
        else:
           x1 = prev_coordinate_x
           y1 = prev_coordinate_y
           x2 = x_coordinates[coord_index]
           y2 = y_coordinates[coord_index]
           velocities[coord_index] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
