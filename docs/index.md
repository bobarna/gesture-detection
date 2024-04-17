# Hand Gesture Detection

## Introduction (TODO Justin)

<img src="assets/videos/motivation-tony-stark.gif?raw=true" alt="Motivation, Tony Stark" width="1000"/>

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
# Explaining Context
[1] M. Oudah, A. Al-Naji, and J. Chahl, "Hand Gesture Recognition Based on Computer Vision: A Review of Techniques", Journal of Imaging, vol. 6,  no. 8, p. 73, July 2020
- [Hand Gesture Recognition Based on Computer Vision: A Review of Techniques](https://www.mdpi.com/2313-433X/6/8/73){:target="_blank"}
- This work covers a variety of contributions to the field of gesture detection from prior literature, including using instrumented gloves and haptic technology with physical connection to a computer, gesture detection via computer vision techniques (spanning color-, appearance-, motion-, skeleton-, depth-, and 3D model based approaches), deep learning for gesture detection, and several applications of these techniques. Each of these approaches are described at a high level, including core concepts, challenges, effectiveness, and more. Several other works are cited in the description of each of these methods through a summarization of knowledge approach. The paper also identifies a ‘research gap’ in the field, wherein researchers are putting effort into gesture recognition in a virtual environment as opposed to practical applications such as healthcare, and challenges for ongoing gesture recognition research/projects. This work is relevant to our project because it set the foundation for the options we had to work with to solve our desired problem, and guided us towards a methodology involving deep neural networks and convolutional neural networks, approaches that are both touched on in the paper.

[2] Y. Fang, K. Wang, J. Cheng, and H. Lu. “A Real-Time Hand Gesture Recognition Method”, 2007 IEEE International Conference on Multimedia and Expo, Beijing, China, 2007, pp. 995-998.
- [A Real-Time Hand Gesture Recognition Method](https://ieeexplore.ieee.org/abstract/document/4284820?casa_token=68LJqOOS6DsAAAAA:8J-3JTymo5tRsv66jeLRbUZRkRJtagqqRxoSMPWGXkw9oA59eg4qp3z5WWAsgzvxnD2YvDpw){:target="_blank"}
- To the field of computer vision, this work contributes a specific method for gesture detection, in which a hand is detected and segmented into different areas via color and feature detection, followed by gesture recognition via planar hand posture structures. This approach combines optical flow and color cue in order to accurately track articulated objects (in this case, a hand). The mean colors of the detected hand are then extracted in HSV color space - an approach that suffers in performance when a hand is shown in front of a similarly-colored material, such as wood. The authors then delve into gesture recognition from here, describing it as a ‘difficult and unsolved’ problem at the time of publication - they cite previous works’ attempts to recognize gestures, such as histogram feature distribution models, fanned boosting detection, and scale-space frameworks for image geometric structures detection, before elaborating on their experimental method for detecting six gestures (left-, right-, up-, and down-pointing thumbs, as well as an open hand and closed fist) - which the authors could predict with 0.938 accuracy using a calculated ‘separability value’. This work contributed to our project most significantly by presenting a foundational study for live, camera-based gesture detection, further contextualizing what we were about to attempt. Furthermore, this paper addresses circumstances in which performance of a feature-detection method could degrade rapidly, informing us to use an approach that would not have similar downfalls.

[3] O. Köpüklü et al., “Real-time hand gesture detection and classification using convolutional neural networks”, IEEE International Conference on Automatic Face and Gesture Recognition, 2019.
- [Real-time hand gesture detection and classification using convolutional neural networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8756576){:target="_blank"}
- This paper’s contribution is a detector consisting of a lightweight convolutional neural network (CNN) architecture to detect gestures, and a classifier which consists of a deep CNN to classify the detected gestures. The authors highlight the practicality of image detection over wearables for users before describing their CNN architecture at a high level. The authors summarize related work regarding using 2D CNNs for information extractions from video data and static images, and proposed 3D CNNs for the extraction of ‘discriminative features’ and dimensions. The methodology for this approach is explained, where feature detection is used to separate between ‘gesture’ and ‘no gesture’ classes, followed by the use of a 3D CNN for gesture classification and post-processing. The authors also describe their experiment to evaluate the performance of this model in offline results against two datasets, EgoGesture and nvGesture. This work is relevant to our project because it was the closest resource we could find that correlated to the architecture we wanted to use, which also incorporates a deep neural network and convolutional neural network.

[4] M. Abavisani et al., “Improving the performance of unimodal dynamic hand gesture recognition with multimodal training”, IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 1165-1174
- [Improving the performance of unimodal dynamic hand gesture recognition with multimodal training](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_HandGesture_Recognition_With_Multimodal_CVPR_2019_paper.pdf){:target="_blank"}
- This work’s contribution is a new framework which leverages the knowledge from multimodal data during training and improves the performance of a unimodal system during testing. In this work, the authors highlight the benefits of training a 3D CNN for each modality data for the recognition of dynamic hand gestures in order to derive a ‘common understanding’ for the contents of these modalities. Related work is covered in the fields of dynamic hand gesture recognition, including pre-existing feature detection and 3D-CNN based hand gesture recognition. The topics of transfer learning and multimodal fusion are briefly covered, before the authors describe their ‘one-3D-CNN-per-modality’ methodology in detail. The results of their experiments with multiple datasets and implementation details using this method are described, before concluding with summary and implications/future directions for continued research. This resource was relevant to our study because it further described the technical details of a model utilizing a convolutional neural network, like ours, and highlighted the validity and feasibility of such an approach.

[5] C. Lugaresi et al., "Mediapipe: A framework for building perception pipelines", Google Research, 2019.
- [Mediapipe: A framework for building perception pipelines](https://arxiv.org/pdf/1906.08172.pdf){:target="_blank"}
- This article summarizes the contributions of MediaPipe, a Python tool which employs a deep neural network-based model trained on a large dataset of hand images to detect and localize landmarks on one's hand, in a high-level way. MediaPipe incorporates additional processing steps and inference models to improve perception technology. This task can be difficult because of excessive coupling between steps - however, MediaPipe these challenges by abstracting and connecting individual perception models into maintainable pipelines. This work relates to our project by providing us the framework to build upon for the additional features of our project, including our own gesture recognition and user interaction components. The landmarking provided by MediaPipe became an essential part of our system.

# Our Project in Context
To our knowledge, our system is the first program designed specifically to allow a user to run real-time gesture detection informed by both a deep neural network for hand-based landmarking and a convolutional neural network for gesture detection on their personal device with an attached simulation element influenced by the gesture detected. Though this problem is specific, this application of gesture recognition was absent in prior literature, and our approach resultantly fills that hole. Our work also places an emphasis on speed of inference over other factors, such as number of recognized gestures, an approach starkly different from other gesture recognition approaches. Thirdly, we emphasized usability and interaction by the user. Though gesture recognition is an important component of our system, we aimed to also visualize an application of these techniques to anyone interacting with the program through the implementation of a fluid simulator. Given that prior literature was largely focused on covering hand gesture approaches at a high level, we assert that this is another open niche which our project fills.

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
Since we use MediaPipe to detect the interest points of the hand, we can use those interest points as the input to a neural network rather than the entire image. This allows us to focus our attention to gestures of the hand, rather than also having to find the hand from an image and then still identify a gesture. MediaPipe outputs 21 interest points identified in 3D space, (x, y, z). This gives us 63 data points to feed into our network. In order to keep inference time low, we use a relatively small model, only 3 layers deep with hidden layer size 256. We use ReLU activation functions between the layers and use SoftMax on the output. Our loss function is cross entropy and the optimizer we chose was stochastic gradient descent. 


#### Fluid Simulation (TODO Barney)

### Contribution (TODO Joseph)
We expect this approach to work better than what has been done before because the specific sequence of deep-learning informed landmarking, convolutional neural network gesture detection, and fluid simulation visualization has not been attempted in prior work. Though we use pieces of methods from pre-existing gesture detection literature, as well as the MediaPipe framework for hand landmarking, the overall system that we have designed is novel. Our project builds on top of this framework to very specifically accomplish our task of, firstly, detecting and landmarking a user’s hand in real time, and secondly, facilitate the detection of specific gestures to trigger corresponding effects in simulation. Given that MediaPipe’s base function is solely recognition of hands in frame and, with the proper options enabled, gesture detection, the functionality we add transforms our system into one no one has tried. Thus, our main contribution is this unique combination of hand gesture recognition techniques alongside user interaction components.

### Intuition (TODO Joseph)
We expect our approach to work to solve the limitation of our related works simply because it has been designed to do so, to most optimally solve the task at hand. This system has been incrementally designed with multiple layers to solve the required independent sub-tasks, and we ensured that each layer worked prior to implementing the next. When it came to detecting and landmarking the hands of the user, we imported the MediaPipe framework and ensured that it could be run locally and accurately display the 21 landmarks prior to implementing further functionality. We similarly implemented our convolutional neural network, gesture detection, and simulation layers, testing incrementally and ensuring the features worked prior to carrying on to the next. This process ensured that these layers were assembled to effectively solve the limitation of the prior work, to have a deep-learning, CNN-informed model with built-in user interaction.

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
Prior approaches to gesture detection are generally more generalizable than ours in terms of number of gestures they are capable of detecting - many systems exist that are specifically designed to detect a wide range of hand signs, while ours is limited to a comparatively small number, due to the relatively niche nature of our goal task. However, in gauging performance of our program, we used the inference time of our system as a success metric. We compared the time taken for hand detection and landmarking of our system and MediaPipe’s built-in gesture recognition option - in doing so, we found that our system generally performs faster for the specific task that we have set out to implement (our average inference time was [INSERT TIME HERE], while the average MediaPipe inference time was [INSERT TIME HERE]). Based on these values, we assert that, despite the relative superiority of other systems in terms of the number of gestures detected, ours performs well against the MediaPipe baseline when detecting the gestures we have implemented and, subsequently, performing functions within the fluid simulator.

### Key Result Presentation (TODO Barney + record a video) 

<video width="50%" style="display: block; margin: 0 auto" controls>
  <source src="assets/videos/real-time-demo-min.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Key Result Performance
## Attempted Approaches
# Hard Coding: 
We tested a couple of different hand gestures to see if the bent-extended detection works well. Here we put a photo to prove that the detection ends up correctly. The print out result is in this format: “right hand shape: ['b', 'e', 'e', 'e', 'e']”, where the ‘e’ represents ‘extended’, and ‘b’ represents ‘bent’, and the five values are corresponding to the thumb, the index finger, the middle finger, the ring finger, and the pinky, respectively.

<img src="assets/images/thumb_bent.png?raw=true" alt="Thumb Bent" width="500"/>

# Recognition
To experiment with detecting shapes, we designed a neural network to detect whether the hand was open in a palm or closed in a fist. The accuracy it achieves for our dataset is 98%. Using this model with MediaPipe we can see the predictor is able to accurately track the hand and the shape it is in. We tried several variations for different aspects of the model, like size, output layer, optimizer, and loss function. We landed on the model described in the methods section because it was able to not just accurately pick the correct shape, but the softmax output also showed it chooses the correct shape with high probability. This is good for our use case because when the hand is not in a fist or a palm, the probabilities given by the model will be low and we can handle cases when the user is not showing a palm or fist. This was also a factor in how we decided what shapes were best to detect. These shapes are quite different which helps the model distinguish between them.

<img src="assets/images/justin-gestures.png?raw=true" alt="Recognition" width="1000"/>

# Simulation

# Velocity
One approach that we attempted in the early- to middle-stages of the project workflow was the calculation of velocity between frames to better inform the dynamics of the fluid simulation. This system was implemented in order to tie into the element of user interaction that we were attempting to build, in order to inform the fluid simulation and the behavior of the movement of the particles (e.g., if a user moved their hand quickly across the screen from left to right, the on-screen fluid would move from left to right at the speed that the hand moved). The way this functionality was implemented was to calculate the Euclidean distance between two subsequent frames with detected hands for each of the 21 landmarks mapped out (42 in the case of two hands being in frame). The velocity for each landmark would automatically be set to zero if there was no prior frame with a detected hand. This velocity-calculation system worked well for calculating the velocity between hands, but when it came to implementing interaction with the fluid simulator, this functionality proved somewhat incompatible with the sim.stable_fluid library, and thus was shelved to make the needed progress towards accomplishing the task we set forth to solve.

## Final Results
We believe that our approach worked. In the operation of the program, we are able to consistently and accurately engage with the different functionalities of the fluid simulator with the landmarks and gesture detection protocols that we have implemented.


## Discussion (TODO together on Wednesday)
## Challenges  (TODO together on Wednesday)
## Contributions (TODO together on Wednesday)

| Member                       | Barnabas Borcsok | James Yu | Joseph Hardin | Justin Wit |
|:-----------------------------|:-----------------|:---------|:--------------|:-----------|
| Interest Points Detection    | V                |          |               |            |
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
