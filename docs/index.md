# Hand Gesture Detection

## Introduction (TODO Justin)


### High Level Description and Motivation
People have a unique ability to have conversations with each other without ever saying a word. Hand gestures are regularly used as quick nonverbal forms of communication to convey things like “good job” with a thumbs up, or “hello” with a wave. Moreover, we are able to have much more meaningful hand gestures as well. Think of an airplane or a cargo ship being directed by a person on the ground. With just a few hand motions, the person directing can communicate with the driver how to steer this giant vehicle exactly where it needs to be. Sometimes words are not the right tool for communicating something, especially when it comes to motion. 

<img src="assets/images/aircraft.png?raw=true" alt="Aircraft Director" width="1000"/>

For example, the person directing the aircraft would have difficulty describing with words to the pilot how to move the plane. Although it is possible, it is much easier to use hand motion to explain how the plane should move, and the results will be much better. Communicating motion to our computers can have similar benefits for how we interact with them, however, computers are much less capable of understanding what we mean by our motion. 

### Specific Problem Definition
This is the issue our project aims to address; how can we help computers understand what we mean with our hand gestures. We are developing a system that can take hand gestures and convey the meaning of that motion to the computer. This involves detecting different hand shapes and tracking the movement of those hand shapes. This type of human-machine interaction will allow motions to be described to a computer in a natural and precise way. There are many applications for this, but for our project we are trying to control the physics of a simulator. So by taking in inputs given by hand gestures, we want to be able to apply motion to the simulator. This requires real time detection and interpretation of motion to allow the user to interact with the system in a way that communicating by other modes cannot achieve. 

### Visuals
<img src="assets/images/intro_vis.png?raw=true" alt="Method Project Outline" width="1000"/>

The image above shows the high level architecture for our project. The user will provide a hand gesture as input to our model. This model is where the bulk of our work lies and involves detecting, interpreting, and describing the input in a way the simulator can understand. We will then feed the output of our model to control some movement in a simulator. 



## Related Work (TODO Joseph)

## Our Method

### Method Overview

#### Recognition (TODO James)

#### Hard Coded Detection (TODO James)

#### Data-Driven Detection (TODO Justin)
Since we use MediaPipe to detect the interest points of the hand, we can use those interest points as the input to a neural network rather than the entire image. This allows us to focus our attention to gestures of the hand, rather than also having to find the hand from an image and then still identify a gesture. MediaPipe outputs 21 interest points identified in 3D space, (x, y, z). This gives us 63 data points to feed into our network. In order to keep inference time low, we use a relatively small model, only 3 layers deep with hidden layer size 256. We use ReLU activation functions between the layers and use SoftMax on the output. Our loss function is cross entropy and the optimizer we chose was stochastic gradient descent. 


#### Fluid Simulation (TODO Barney)

### Contribution (TODO Joseph)

### Intuition (TODO Joseph)

### Visuals (TODO Justin)

<img src="assets/images/pipelinevis.png?raw=true" alt="Methods Pipeline" width="1000"/>
The flow chart above shows the major steps of our process, including our use of MediaPipe, neural network classification, and outputting to the simulator.

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
