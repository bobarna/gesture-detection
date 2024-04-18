# Hand Gesture Detection

## Introduction (TODO Justin)

<video width="50%" style="display: block; margin: 0 auto" controls>
  <source src="assets/videos/real-time-demo-min.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

### High Level Description and Motivation
People have a unique ability to have conversations with each other without ever saying a word. Hand gestures are regularly used as quick nonverbal forms of communication to convey things like “good job” with a thumbs up, or “hello” with a wave. Moreover, there are hand gestures in use for critical systems. Think of an airplane or a cargo ship being directed by a person on the ground. With just a few hand motions, the person directing can communicate with the driver how to steer this giant vehicle exactly where it needs to be. Sometimes words are not the right tool for communicating something, especially when it comes to motion. 

<img src="assets/images/aircraft.png?raw=true" alt="Aircraft Director" width="1000"/>
<center><em>Figure 1</em></center>

For example, the person directing the aircraft would have difficulty describing with words to the pilot how to move the plane. Although it is possible, it is much easier to use hand motion to explain how the plane should move, and the results will be much better. Communicating motion to our computers can have similar benefits for how we interact with them, however, computers are much less capable of understanding what we mean by our motion. 

### Specific Problem Definition
This is the issue our project aims to address; how can we help computers understand what we mean with our hand gestures. We are developing a system that can take hand gestures and convey the meaning of that motion to the computer. This involves detecting different hand shapes and tracking the movement of those hand shapes. This type of human-machine interaction will allow motions to be described to a computer in a natural and precise way. There are many applications for this, but for our project we are trying to control a physics-based simulation. By taking inputs in the form of hand gestures, we want to apply motion to this simulator. This requires real time detection and interpretation of motion to allow the user to interact with the system in a way that communicating by other modes cannot achieve. 

### Visuals
<img src="assets/images/intro_vis.png?raw=true" alt="Method Project Outline" width="1000"/>
<center><em>Figure 2</em></center>

The image above shows the high level architecture for our project. The user will provide a hand gesture as input to our model. This model is where the bulk of our work lies and involves detecting, interpreting, and describing the input in a way the simulator can understand. We will then feed the output of our model to control some movement in a simulator. 

## Related Work (TODO Joseph)
### Explaining Context
[1] M. Oudah, A. Al-Naji, and J. Chahl, "Hand Gesture Recognition Based on Computer Vision: A Review of Techniques", Journal of Imaging, vol. 6,  no. 8, p. 73, July 2020
- [Hand Gesture Recognition Based on Computer Vision: A Review of Techniques](https://www.mdpi.com/2313-433X/6/8/73)
- This work covers a variety of contributions to the field of gesture detection, including using instrumented gloves and haptic technology with physical connection to a computer, gesture detection via computer vision techniques (spanning color-, appearance-, motion-, skeleton-, depth-, and 3D model based approaches), deep learning for gesture detection, and several applications of these techniques. Each of these approaches are described at a high level, including core concepts, challenges, effectiveness, and more. This work is relevant to our project because it set the foundation for the options we had to work with to solve our desired problem, and guided us towards a methodology involving multiple layers of neural networks, an approach that was touched on in the paper. Additionally, the requirement for specialized equipment associated with many of these approaches informed our desire to create a non-specialized approach that any user could easily run on their personal device.

[2] Y. Fang, K. Wang, J. Cheng, and H. Lu. “A Real-Time Hand Gesture Recognition Method”, 2007 IEEE International Conference on Multimedia and Expo, Beijing, China, 2007, pp. 995-998.
- [A Real-Time Hand Gesture Recognition Method](https://ieeexplore.ieee.org/abstract/document/4284820?casa_token=68LJqOOS6DsAAAAA:8J-3JTymo5tRsv66jeLRbUZRkRJtagqqRxoSMPWGXkw9oA59eg4qp3z5WWAsgzvxnD2YvDpw)
- To the field of computer vision, this work contributes a specific method for gesture detection, in which a hand is detected and segmented into different areas via color and feature detection, followed by gesture recognition via planar hand posture structures. This approach combines optical flow and color cue in order to accurately track articulated objects (in this case, a hand). The mean colors of the detected hand are then extracted in HSV color space - an approach that suffers in performance when a hand is shown in front of a similarly-colored material, such as wood. The authors then delve into gesture recognition from here, describing it as a ‘difficult and unsolved’ problem at the time of publication - they cite previous works’ attempts to recognize gestures, such as histogram feature distribution models, fanned boosting detection, and scale-space frameworks for image geometric structures detection, before elaborating on their experimental method for detecting six gestures (left-, right-, up-, and down-pointing thumbs, as well as an open hand and closed fist) - which the authors could predict with 0.938 accuracy using a calculated ‘separability value’. This work contributed to our project most significantly by presenting a foundational study for live, camera-based gesture detection, further contextualizing what we were about to attempt. Furthermore, this paper addresses circumstances in which performance of a feature-detection method could degrade rapidly, informing us to use an approach that would not have similar downfalls.

[3] O. Köpüklü et al., “Real-time hand gesture detection and classification using convolutional neural networks”, IEEE International Conference on Automatic Face and Gesture Recognition, 2019.
- [Real-time hand gesture detection and classification using convolutional neural networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8756576)
- This paper’s contribution is a detector consisting of a lightweight convolutional neural network (CNN) architecture to detect gestures, and a classifier which consists of a deep CNN to classify the detected gestures. The authors highlight the practicality of image detection over wearables for users before describing their CNN architecture at a high level. The authors summarize related work regarding using 2D CNNs for information extractions from video data and static images, and proposed 3D CNNs for the extraction of ‘discriminative features’ and dimensions. The methodology for this approach is explained, where feature detection is used to separate between ‘gesture’ and ‘no gesture’ classes, followed by the use of a 3D CNN for gesture classification and post-processing. The authors also describe their experiment to evaluate the performance of this model in offline results against two datasets, EgoGesture and nvGesture. This work is relevant to our project because it was the closest resource we could find that correlated to the architecture we wanted to use, which also incorporates a multiple neural networks for gesture recognition.

[4] M. Abavisani et al., “Improving the performance of unimodal dynamic hand gesture recognition with multimodal training”, IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 1165-1174
- [Improving the performance of unimodal dynamic hand gesture recognition with multimodal training](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_HandGesture_Recognition_With_Multimodal_CVPR_2019_paper.pdf)
- This work’s contribution is a framework which leverages the knowledge from multimodal data during training and improves the performance of a system through transfer learning, in which positive knowledge from feature detection is propagated and negative knowledge is suppressed. In this work, the authors highlight the benefits of training a 3D CNN for the recognition of dynamic hand gestures. Related work is covered in the fields of dynamic hand gesture recognition, including pre-existing feature detection and 3D-CNN based hand gesture recognition. The results of the authors' experiments with multiple datasets and implementation details using ththe authors' method are described, before concluding with summary and implications/future directions for continued research. This resource was relevant to our study because it further described the technical details of a model utilizing neural networks, like ours, and highlighted the validity and feasibility of such an approach.

[5] C. Lugaresi et al., "Mediapipe: A framework for building perception pipelines", Google Research, 2019.
- [Mediapipe: A framework for building perception pipelines](https://arxiv.org/pdf/1906.08172.pdf)
- MediaPipe is an on-device Machine Learning solution from Google. For gesture detection, it employs a deep neural network-based model to detect and localize landmarks on one's hand. MediaPipe incorporates additional processing steps in the background to deliver state-of-the-art results. MediaPipe contributes to the field by providing an extendable and maintainable pipeline for developers to build on. We incorporate MediaPipe into our gesture recognition pipeline for detecting hand landmarks in image space, making it an essential part of our system.

## Our Project in Context
To our knowledge, our system is the first program designed specifically to allow a user to run real-time gesture detection informed by both a deep neural network for hand-based landmarking and a separately layered neural network for gesture detection on their personal device with an attached simulation element influenced by the gesture detected. Though this problem is specific, this application of gesture recognition was absent in prior literature, and our approach resultantly fills that hole. Our work also places an emphasis on speed of inference over other factors, such as number of recognized gestures, an approach starkly different from other gesture recognition approaches. Thirdly, we emphasized usability and interaction by the user. Though gesture recognition is an important component of our system, we aimed to also visualize an application of these techniques to anyone interacting with the program through the implementation of a fluid simulator. Given that prior literature was largely focused on covering hand gesture approaches at a high level, we assert that this is another open niche which our project fills.

## Our Method

### Method Overview
We will now cover the components of our current best approach. This can be divided into four sequential segments, listed below: the recognition of hand gestures via MediaPipe, the hard coded values by which we rapidly identify finger position based on joint angles, the neural network-based approach for gesture identification, and the connection to the system's interactive element, the fluid simulator.

#### Recognition (TODO James)
We use MediaPipe's deep neural network-based models to detect and localize hand
landmarks. The machine learning pipeline consists of several models working
together:
  - A palm detector model is used to find an oriented hand bounding box.
  - A hand landmark model returns the 3D hand keypoints in the image region found by the palm detector model.

The output of MediaPipe contains 21 keypoints (as the following graph shows),
covering the 5 fingers of a hand, and it also outputs whether the detected hand
is left or right.

<img src="assets/images/hand_landmarks.png?raw=true" alt="Hand Landmarks" width="1000"/>
<center><em>Figure 3</em></center>

#### Hard Coded Detection (TODO James)
We obtain hand gestures by detecting whether each finger is bent or extended.
This can be done by calculating the angle of each finger knuckle. First, we
get the vectors between the keypoints returned by MediaPipe, then get
the cosine of the angle by calculating the dot product. After testing, we set
the threshold between bend and extend to be 0.9. That is, if all the knuckles on
a finger has cosine theta bigger than 0.9, then this finger is straight; while
if any of the knuckles have cosine theta less than 0.9, meaning that this knuckle
is bent, then the finger will be classified as bent. Detecting the shape of each
finger can help us distinguish many different kinds of hand gestures.

#### Data-Driven Detection (TODO Justin)
Since we use MediaPipe to detect the interest points of the hand, we can use those interest points as the input to a neural network rather than the entire image. This allows us to focus our attention to gestures of the hand, rather than also having to find the hand from an image and then still identify a gesture. MediaPipe outputs 21 interest points identified in 3D space, (x, y, z). This gives us 63 data points to feed into our network. In order to keep inference time low, we use a relatively small model, only 3 layers deep with hidden layer size 256. We use ReLU activation functions between the layers. Our loss function is cross entropy and the optimizer we chose was Adam. 

#### Fluid Simulation
We implement a simple fluid simulation based on [Stable Fluids](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf) by Jos Stam. We enjoy the parallelism and device acceleration offered by [Taichi](https://www.taichi-lang.org/), while keeping our code portable. Our 2D simulation domain is discretized on a 512x512 grid, same as the resolution we use for detecting the keypoints. For performance considerations, apart from the grid resolution, tweaking the number of jacobi iterations used in the pressure solve offers a tradeoff between quality and performance. Depending on the user's device, a lower grid resolution and/or a lower number of jacobi iterations can offer faster computation speed. Further, the gravity coefficient of the simulation can also be set by the user interactively, by bending the thumb and/or pinky fingers, while keeping the rest of the fingers extended.

### Contribution (TODO Joseph)
We expect this approach to work better than what has been done before because the specific approach of deep-learning informed landmarking, neural network gesture detection, and fluid simulation visualization has not been attempted in prior work in a way that focuses on accessibility and speed for average users. Though we use pieces of methods from pre-existing gesture detection literature, as well as the MediaPipe framework for hand landmarking, the way we have approached the problem is, on the whole, unique from other preexisting approaches. Our project builds on top of these preexisting frameworks to very accomplish our task of, firstly, detecting and landmarking a user’s hand in real time, and secondly, facilitate the detection of specific gestures to trigger corresponding effects in simulation in real time in an accessible manner. Given that MediaPipe’s base function is recognition of hands in frame and gesture detection (with the proper settings), the methodology we use and the functionality we add focusing on accessibility and speed makes our approach the most optimal for a user-friendly gesture detection experience. Thus, our main contribution is this unique combination of hand gesture recognition techniques alongside optimized user interaction components for local system use.

### Intuition (TODO Joseph)
We expect our approach to work to solve the limitation of our related works simply because it has been designed to do so, to most optimally solve the task at hand with quick inferences and operations, as well as enable maximum ease of access for system users. This system has been incrementally designed with multiple layers to solve the required independent sub-tasks, and we ensured that each layer worked prior to implementing the next. When it came to detecting and landmarking the hands of the user, we imported the MediaPipe framework and ensured that it could be run locally and accurately display the 21 landmarks prior to implementing further functionality. We similarly implemented our neural network, gesture detection, and simulation layers, testing incrementally and ensuring the features worked prior to carrying on to the next. This process ensured that these layers were assembled to effectively solve the limitation of the prior work, to have a deep-learning, neural network-informed model with built-in, optimized user interaction.

### Visuals (TODO Justin)

<img src="assets/images/pipelinevis.png?raw=true" alt="Methods Pipeline" width="1000"/>
<center><em>Figure 4</em></center>
The flow chart above shows the major steps of our process, including our use of MediaPipe, neural network classification, and outputting to the simulator.

## Experiments
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

### Experiment Purpose
We grab the camera stream from the user's device, and detect interest points for
the user's hands. Based on these, we let the user interact with a colorful fluid
simulation in real time.

### Input Description

<img src="assets/images/input.jpg?raw=true" alt="Input" width="1000"/>
<center><em>Figure 5</em></center>

Figure 5 shows that each input image is a frame of a user standing in front of a camera. But this isn't the input to our neural network, nor is it the input to our simulation. We use the landmarks identified by MediaPipe as the input to our network to identify shape. This means we had to collect our own dataset of landmarks for the shapes we wanted to detect. We used 1000 sets of landmarks for each shape to train our network. We then use the output of our network in combination with our velocity calculations and individual finger geometry calculations as our input to the simulator.

### Desired Output Description
The output of our model is an interactive real-time
fluid simulation. We ended up experimenting with blending the fluid simulation
with the current frame from the camera for artistic reasons, but visualizing
only the fluid simulation with the detected keypoints also delivered a pleasant
user experience.

### Metric for Success
Our metric for success is two fold and starts with the model's prediction accuracy and loss. Having a high accuracy and low loss is important since we want to be confident in the shapes we are detecting. But our end goal isn't just a good classifier of hand gestures, we want success in using the gestures in a simulator. Metrics like ease of use, robustness, and system lag will show how effective our solution is for communicating motion. During our qualitative analysis, our users enjoyed a real-time experience of interacting with the system. Even when trying extreme motions, the fluid simulation remained stable, even when detection quality slightly varied. Please see the video below for a recording of a user interacting with our system.

## Results
### Baselines (TODO Joseph)
Prior approaches to gesture detection are generally more generalizable than ours in terms of number of gestures they are capable of detecting - many systems exist that are specifically designed to detect a wide range of hand signs, while ours is limited to a comparatively small number, due to the relatively niche nature of our goal task. However, in gauging performance of our program, we used the inference time of our system as a success metric. We compared the time taken for hand detection and landmarking of our system and MediaPipe’s built-in gesture recognition option - in doing so, we found that our system generally performs faster for the specific task that we have set out to implement (our average inference time was 3.51 ms, while the average MediaPipe inference time was 43.6 ms). Based on these values, we assert that, despite the relative superiority of other systems in terms of the number of gestures detected, ours performs well against the MediaPipe baseline when detecting the gestures we have implemented and, subsequently, performing functions within the fluid simulator.

### Key Result Presentation (TODO Barney + record a video) 

<video width="50%" style="display: block; margin: 0 auto" controls>
  <source src="assets/videos/real-time-demo-min.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Key Result Performance
#### Attempted Approaches
##### Hard Coding 
We tested a couple of different hand gestures to see if the bent-extended detection works well. Here we put a photo to prove that the detection ends up correctly. The print out result is in this format: “right hand shape: ['b', 'e', 'e', 'e', 'e']”, where the ‘e’ represents ‘extended’, and ‘b’ represents ‘bent’, and the five values are corresponding to the thumb, the index finger, the middle finger, the ring finger, and the pinky, respectively.

<img src="assets/images/thumb_bent.png?raw=true" alt="Thumb Bent" width="1000"/>
<center><em>Figure 6</em></center>

##### Recognition
To experiment with detecting shapes, we designed a neural network to detect whether the hand was open in a palm or closed in a fist. The accuracy it achieves for our dataset is 98%. 

<img src="assets/images/trainingplot.png?raw=true" alt="Test and Train Plots" width="1000"/>
<center><em>Figure 7</em></center>

Using this model with MediaPipe we can see the predictor is able to accurately track the hand and the shape it is in. We tried several variations for different aspects of the model, like size, output layer, optimizer, and loss function. We landed on the model described in the methods section because it was able to not just accurately pick the correct shape, but the softmax output also showed it chooses the correct shape with high probability. This is good for our use case because when the hand is not in a fist or a palm, the probabilities given by the model will be low and we can handle cases when the user is not showing a palm or fist. This was also a factor in how we decided what shapes were best to detect. These shapes are quite different which helps the model distinguish between them.

<img src="assets/images/justin-gestures.png?raw=true" alt="Recognition" width="1000"/>
<center><em>Figure 8</em></center>

##### Velocity
One approach that we attempted in the early- to middle-stages of the project workflow was the calculation of velocity between frames to better inform the dynamics of the fluid simulation. This system was implemented in order to tie into the element of user interaction that we were attempting to build, in order to inform the fluid simulation and the behavior of the movement of the particles (e.g., if a user moved their hand quickly across the screen from left to right, the on-screen fluid would move from left to right at the speed that the hand moved). The way this functionality was implemented was to calculate the Euclidean distance between two subsequent frames with detected hands for each of the 21 landmarks mapped out (42 in the case of two hands being in frame). The velocity for each landmark would automatically be set to zero if there was no prior frame with a detected hand. This velocity-calculation system worked well for calculating the velocity between hands, but when it came to implementing interaction with the fluid simulator, this functionality proved somewhat incompatible with the sim.stable_fluid library, and thus was shelved to make the needed progress towards accomplishing the task we set forth to solve. 

#### Final Results
We believe that our approach worked. In the operation of the program, we are able to consistently and accurately engage with the different functionalities of the fluid simulator with the landmarks and gesture detection protocols that we have implemented.

## Discussion (TODO together on Wednesday)
To begin with our accomplishments, we all were able to meet the goal we set at the beginning of this project of creating an easily accessible system in which a user could utilize hand gestures to interact with a virtual element, which we implemented in the form of the fluid simulator.
Additionally, this project opened all of our eyes to the wide array of tasks related to human-machine interaction in the field of computer vision. These motions that feel simple and intuitive to us are, by contrast, incredibly difficult for a computer to understand, and it is correspondingly challenging to describe these gestures in a useful way to a computer. We were also able to learn more about some of the concepts that we had been exposed to in lecture through the construction of our system, such as feature detection, the usage of neural networks, and more. Furthermore, we were all able to practice more logistical factors of the project, such as version control with Github, maintaining regular communication to meet self-imposed deadlines, and task delegation. Much remains to be done for future work. Through the approach that we implemented for this assignment, we were able to introduce the concept of making a priority out of accessibility and speed for gesture detection systems - however, there are likely ways that a system such as ours could be further optimized for quick operation and ease of access, along with incorporating a wider variety of recognizable gestures, to create a more functionally robust and user-friendly application. There is a long way to go for a system like ours to be as robust as Tony Stark's holograms (see below).

## Challenges  (TODO together on Wednesday)
Though our team worked well together, there were a couple of challenges that we ran into as we developed our project. One such challenge was identifying which gestures we wanted to describe to the computer. Though our initial vision for the project was to implement a wide variety of gestures (or even a sequence of gestures for continuous motion), we quickly realized that this scope of work was likely too complex to achieve the inference speed we wanted to attain. Thus, we had to settle for the neural network detection of the 'fist' and 'palm' gestures. Though these gestures were enough for our purposes, and conducive to the factors of speed and accessibility that we were aiming for, further work to improve functionality to allow a system like ours to detect these complex gestures while preserving the inference speed and ease of use would be very beneficial for an even more immersive gesture-based interaction experience. In the same vein, we could have benefitted from locking down an appropriate scope of work earlier in the lifespan of the project. We had several great ideas for what we wanted the final project to look like and the features we wanted to implement. However, it would have taken much longer than one semester to appropriately and optimally implement all of these ideas into one program, and we thus may have benefitted from focusing our efforts even sooner than we did. If we were to start over today, a different strategy we could take to make progress on our overall problem statement would be to mitigate these two factors - quickly identifying which gestures would be most appropriate to enact for our purposes as soon as we could, and clearly laying out a feasible scope of work to identify our starting point, checkpoints, and stopping points from the beginning.


<img src="assets/videos/motivation-tony-stark.gif?raw=true" alt="Motivation, Tony Stark" width="1000"/>

## Contributions

| Member                       | Barney Börcsök     | James Yu           | Joseph Hardin      | Justin Wit         |
|:-----------------------------|:-------------------|:-------------------|:-------------------|:-------------------|
| MediaPipe Setup              | &#10003;           |                    |                    |                    |
| Neural Network               |                    |                    |                    | &#10003;           |
| Finger Extend or Bend        |                    | &#10003;           |                    |                    |
| Velocity (First iteration)   |                    |                    | &#10003;           |                    |
| Velocity (Second iteration)  | &#10003;           |                    |                    | &#10003;           |
| Fluid Simulation             | &#10003;           |                    |                    |                    |
| Merging the code             | &#10003;           | &#10003;           | &#10003;           | &#10003;           |
| Code cleanup                 |                    |                    | &#10003;           |                    |
| Combining Techniques         | &#10003;           | &#10003;           | &#10003;           | &#10003;           |
| Related Works                |                    |                    | &#10003;           |                    |
| Writing the report           | &#10003;           | &#10003;           | &#10003;           | &#10003;           |
| Github Setup                 | &#10003;           |                    |                    |                    |
| Github Page                  |                    | &#10003;           |                    | &#10003;           |

