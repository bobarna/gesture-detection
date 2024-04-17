# Hand Gesture Detection

## Introduction (TODO Justin)

### High Level Description and Motivation

### Specific Problem Definition

### Visuals

TODO

## Related Work (TODO Joseph)

## Our Method

### Method Overview

#### Recognition (TODO James)
- MediaPipe utilizes deep neural network-based models trained on a large dataset of hand images to detect and localize the hand landmarks. The machine learning pipeline consists of several models working together:
    - A palm detector model is used to find an oriented hand bounding box.
    - A hand landmark model returns the 3D hand keypoints in the image region found by the palm detector model.
    - A gesture recognizer classifies the previously obtained keypoint structure into a discrete set of gestures.
- The output of MediaPipe contains 21 keypoints (as the following graph shows), covering the 5 fingers of a hand, and it also outputs whether the detected hand is left or right.
<img src="assets/images/hand_landmarks.png?raw=true" alt="Hand Landmarks" width="1000"/>

#### Hard Coded Detection (TODO James)
We obtain hand gestures by detecting whether each finger is bent or extended. This work can be done by calculating the angle of each finger knuckle. We firstly get the vectors between the keypoints returned by MediaPipe, then get the cosine of the angle by calculating the dot product. After testing, we set the threshold between bend and extend to be 0.9. That is, if all the knuckles on a finger has cosine theta bigger than 0.9, then this finger is straight; while if any of the knuckles has cosine theta less than 0.9, meaning that this knuckle is bent, then the finger will be classified as bent. Detecting the shape of each finger can help us distinguish many different kinds of hand gestures.

#### Data-Driven Detection (TODO Justin)

#### Fluid Simulation (TODO Barney)

### Contribution (TODO Joseph)

### Intuition (TODO Joseph)

### Visuals (TODO Justin)

## Experiments (TODO Barney)
### Experiment Purpose (TODO Barney)
### Input Description (TODO Barney)
### Desire Output Description (TODO Barney)
### Metric for Success (TODO Barney)

## Results
### Baselines (TODO Joseph)
### Key Result Presentation (TODO Barney + record a video) 
### Key Result Performance
#### Hard Coding: 
We tested a couple of different hand gestures to see if the bent-extended detection works well. Here we put a photo to prove that the detection ends up correctly. The print out result is in this format: “right hand shape: ['e', 'e', 'b', 'b', 'e']”, where the ‘e’ represents ‘extended’, and ‘b’ represents ‘bent’, and the five values are corresponding to the thumb, the index finger, the middle finger, the ring finger, and the pinky, respectively.
<img src="assets/images/thumb_bent.png?raw=true" alt="Thumb Bent" width="1000"/>

## Discussion (TODO together on Wednesday)
## Challenges  (TODO together on Wednesday)
## Contributions (TODO together on Wednesday)

| Member                       | Barnabas Borcsok | James Yu | Joseph Hardin | Justin Wit |
|:-----------------------------|:-----------------|:---------|:--------------|:-----------|
| Interest Points Detection | V                |          |               |            |
| MediaPipe Setup              | V                |          |               |            |
| Neural Network               |                  |          |               | V          |
| Finger Extend or Bend        |                  | V        |               |            |
| Velocity Detection           |                  |          | V             |            |
| Introduction                 | V                | V        |               | V          |
| Related Works                |                  | V        | V             |            |
| Methods                      | V                | V        | V             | V          |
| Experiments                  |                  | V        | V             | V          |
| What's Next                  |                  |          |               | V          |
| Github Setup                 | V                |          |               |            |
| Github Page                  |                  | V        |               | V          |
| Detection Model              | V                | V        |               |            |
| Hand Velocity                |                  |          | V             | V          |
| Combining Techniques         | V                | V        | V             | V          |
| Application                  | V                | V        | V             | V          |
