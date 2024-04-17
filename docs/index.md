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

#### Hard Coded Detection (TODO James)

#### Data-Driven Detection (TODO Justin)

#### Fluid Simulation (TODO Barney)

### Contribution (TODO Joseph)

### Intuition (TODO Joseph)

### Visuals (TODO Justin)

## Experiments (TODO Barney)
We implemented our method in 
[Python](https://www.python.org/), 
using [OpenCV](https://opencv.org/),
[PyTorch](https://pytorch.org/) and [Numpy](https://numpy.org/). For the fluid
simulation, we use the [Taichi Language](https://www.taichi-lang.org/),
a high-performance parallel programming language in Python. By using Taichi, we
make our code easily portable between different systems, GPU (or even CPU)
backends, while enjoying the speed benefits of device-accelerated parallel
programming.

We provide source code and instructions on running our code at
[https://github.com/bobarna/gesture-detection](https://github.com/bobarna/gesture-detection).

### Experiment Purpose (TODO Barney)
We grab the camera stream from the user's device, and detect interest points for
the user's hands. Based on these, we let the user interact with a colorful fluid
simulation in real time.

### Input Description (TODO Barney)
Figure XX (TODO) shows that each input image is a frame of a user standing in
front of a camera. As we decided to run our fluid simulation in pixel space,
matching the domain's resolution to our camera image, we decided to crop our
input image to a 512x512 resolution. We observed that this resolution offers an
enjoyable user experience, while enabling real-time performance of our pipeline
even on low-end devices we tested. 

### Desire Output Description (TODO Barney)
Figure XX (TODO) shows that the output of our model is an interactive real-time
fluid simulation. We ended up experimenting with blending the fluid simulation
with the current frame from the camera for artistic reasons, but visualizing
only the fluid simulation with the detected keypoints also delivered a pleasant
user experience. 

### Metric for Success (TODO Barney)
During our qualitative analysis, our users enjoyed a real-time experience of
interacting with the system. Even when trying extreme motions, the fluid
simulation remained stable, even when detection quality slightly varied. We
observe that a perfect detection was not necessary for a good user experience
and end result. Please see the video below for a recording of a user interacting
with our system.

## Results
### Baselines (TODO Joseph)
### Key Result Presentation (TODO Barney + record a video) 

![Real-time Gesture Detection Demo](https://github.com/bobarna/gesture-detection/blob/6f531aba9e2946b6e83f8b7853b5270d420fb6b9/docs/assets/videos/real-time-demo-min.mp4)

### Key Result Performance

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
