## Introduction

People have a unique ability to have conversations with each other without ever saying a word. Hand gestures are regularly used as quick nonverbal forms of communication to convey things like “good job” with a thumbs up, or “hello” with a wave. Moreover, we are able to have much more meaningful hand gestures as well. Think of an airplane or a cargo ship being directed by a person on the ground. With just a few hand motions, the person directing can communicate with the driver how to steer this giant vehicle exactly where it needs to be.  

<img src="docs/assests/images/aircraft.png?raw=true" alt="Aircraft Director" width="1000"/>

Sometimes words are not the right tool for comminating something, especially when it comes to motion. For example, the person directing the aircraft would have difficulty describing with words to the pilot how to move the plane. Although it is possible, it is much easier to use hand motion to explain how the plane should move, and the results will be much better. Communicating motion to our computers faces a similar issue, however, computers are much less capable of understanding what we mean by our motion. 

This is the issue our project aims to address. We are developing a system that can take hand gestures and convey the meaning of that motion to the computer. This involves detecting different hand shapes and tracking the movement of those hand shapes. This type of human-machine interaction will allow motions to be described to a computer in a more natural and more precise way. There are many applications for this, but some specifics we find fascinating are controlling the physics of a simulation and controlling the movement of a robot. These both require real time detection and interpretation of the motion, and allow the user to interact with the system in a way that communicating by other modes cannot achieve.  

## Related Works  

### Section 1: Summaries of Knowledge/Various Techniques  
- [1] M. Oudah, A. Al-Naji, and J. Chahl, “Hand Gesture Recognition Based on Computer Vision: A Review of Techniques,” Journal of Imaging, vol. 6, no. 8, p. 73, Jul. 2020.  
  - https://www.mdpi.com/2313-433X/6/8/73  
  - This work covers a variety of approaches from prior literature, including using instrumented gloves and haptic technology with physical connection to the 
computer, gesture detection via computer vision techniques (spanning color-, appearance-, motion-, skeleton-, depth-, and 3D model based approaches), deep 
learning for gesture detection, and several applications of these techniques. Each of these approaches are described at a high level, including core concepts, 
challenges, effectiveness, and more. Several other works are cited in the description of each of these methods through a summarization of knowledge 
approach. The paper also identifies a ‘research gap’ in the field, wherein researchers are putting effort into gesture recognition in a virtual environment as 
opposed to practical applications such as healthcare, and challenges for ongoing gesture recognition research/projects.  
- [2] S. Mitra and T. Acharya, "Gesture Recognition: A Survey," in IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), vol. 37, no. 3, pp. 311-324, May 2007  
  - https://ieeexplore.ieee.org/abstract/document/4154947?casa_token=TLKPdKUhEAAAAA:phFKQACVGDxS2JXZwkD9BzJeNvuj6YVtqigT2TSlyZDL1_aoNyngYuIcWqSxYT0k1jmxLJCJyA  
  - This work covers a variety of approaches from prior literature, conducting a survey on the field as a whole and covering the specifics of several applications 
utilizing different techniques, as well as highlighting challenges present in the field at the time of publication, and possible directions for future research. The 
paper describes several applications for gesture recognition technology before going in depth about specific tools used for gesture recognition, such as Hidden 
Markov Models, Kalman filtering, particle filtering, condensation articles, and more. Computer vision techniques are touched on more specifically, covering 
feature detection, clustering, image processing, and more. The theory behind these tools are elaborated on technically, before a brief summary of the different types 
of hand and arm gestures, as well as a similar summary of face and head gestures. The authors close this work by summarizing the tools covered and potential 
directions for future approaches that combine aforementioned tools, as well as potential ways to surmount existing difficulties in gesture recognition, such as 
challenges with gesture categorization, transition between expressions, and more.  
- [3] Y. Wu and T. S. Huang, “Vision-based Gesture Recognition: A Review,” Gesture Based Communication in Human-Computer Interaction, pp. 103–115, 1999.  
  - https://link.springer.com/chapter/10.1007/3-540-46616-9_10  
  - This work conducts a survey of methods and applications for vision-based gesture recognition, highlighting the features that can be detected for gesture recognition, such as most discriminating features (MDFs) and most expressive features (MEFs), the application of Hidden Markov Models for detection of 3D features, 
prerequisite data collection, and more. The authors also describe the different categories of gestures before covering the application systems in which gesture 
recognition could be used (virtual environments, sign language translation, etc.). In this summary, the works of previous authors in this realm are covered. The 
importance of feature recognition in gesture recognition is emphasized, and previous approaches for this specifically are described at a high level. The authors 
then discuss approaches for static and temporal gesture recognition, use of HMMs, and more. Sign language detection and future directions are described.  


### Section 2: Solutions Involving Feature Detection  
- [4] Y. Fang, K. Wang, J. Cheng and H. Lu, "A Real-Time Hand Gesture Recognition Method," 2007 IEEE International Conference on Multimedia and Expo, Beijing, China, 
2007, pp. 995-998  
  - https://ieeexplore.ieee.org/abstract/document/4284820?casa_token=68LJqOOS6DsAAAAA:8J-3JTymo5tRsv66jeLRbUZRkRJtagqqRxoSMPWGXkw9oA59eg4qp3z5WWAsgzvxnD2YvDpw  
  - This work covers a specific method for gesture detection, in which a hand is detected and segmented into different areas via color and feature detection, 
followed by gesture recognition via planar hand posture structures. This approach combines optical flow and color cue in order to accurately track articulated 
objects (in this case, a hand). The mean colors of the detected hand are then extracted in HSV color space - an approach that suffers in performance when a 
hand is shown in front of a similarly-colored material, such as wood. The authors then delve into gesture recognition from here, describing it as a ‘difficult and 
unsolved’ problem at the time of publication. They cite previous works’ attempts to recognize gestures, such as histogram feature distribution models, fanned 
boosting detection, and scale-space frameworks for image geometric structures detection, before elaborating on their experimental method for detecting six 
gestures (left-, right-, up-, and down-pointing thumbs, as well as an open hand and closed fist). The authors utilize a calculated ‘separability value’ to detect these 
gestures with 0.938 accuracy.  


### Section 3: Solutions Involving Depth and Density  
- [5] H. Tang et al., “Fast and robust dynamic hand gesture recognition via key frames extraction and feature fusion,” Neurocomputing, Volume 331, pp. 424-433.  
  - https://www.sciencedirect.com/science/article/pii/S0925231218313663  
  - This work combines image entropy and density clustering to exploit the key frames from hand gesture video, improving the efficiency of recognition. The 
authors highlight early in this work that the issues of having gesture recognition be fast and robust are significant and ongoing, largely as a result of prior work 
drawing on using the whole data series (which results in inherently degraded performance). Key frame extraction is proposed as a method by which redundant 
data could be eliminated from the dataset, utilizing ‘image entropy’ to evaluate individual frames, utilizing the peaks of the charted data to find the key frames. 
The authors then describe a framework for hand gesture recognition wherein the key frame methodology is combined with existing methods for feature detection. 
This framework is then tested on two clear, uncluttered, relatively noiseless publicly available datasets as well as two new datasets. This method of key frame 
extraction and feature detection results in higher accuracy than several previous attempts, and the authors claim their method outperforms ‘state-of-the-art’ 
methods on the same datasets.  
- [6] J. Suarez and R. R. Murphy, "Hand gesture recognition with depth images: A review," 2012 IEEE RO-MAN: The 21st IEEE International Symposium on Robot and Human 
Interactive Communication, Paris, France, 2012, pp. 411-417  
  - https://ieeexplore.ieee.org/abstract/document/6343787?casa_token=QvzrzrAY8QYAAAAA:9Z46MbV2WH0l1cgcSgOl9CwGL0tjbGIT1DJHY0gg_LcQmZt-XiLrdQa8v9T07K-6VcQxQuYsQQ  
  - This work focuses specifically on the use of depth as the primary feature used for hand recognition, describing how depth cameras can function in circumstances 
where video cameras may fail, the environments and applications where such methods are being employed for research purposes, and the effect of wide release 
via the Microsoft Kinect (whether the libraries released with the sensor are being used, if they are being replaced with custom algorithms, and more). Depth 
thresholding is specifically covered as a method for isolating hands, highlighted as an advantage of using depth cameras over color cameras for gesture recognition. 
Other hand segmentation methods from previous works are covered at a high level before the authors describe gestures, elaborating on the categories established by 
previous work, and the methods by which they can be classified, such as Hidden Markov Models, k-nearest neighbors, and more.  


### Section 4: Solutions Involving Data Training/Neural Networks  
- [7] O. Köpüklü et al., “Real-time hand gesture detection and classification using convolutional neural networks,” IEEE International Conference on Automatic Face and 
Gesture Recognition, 2019.  
  - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8756576  
  - This paper uses a detector which consists of a lightweight convolutional neural network (CNN) architecture to detect gestures, and a classifier which consists of a 
deep CNN to classify the detected gestures. The authors highlight the practicality of image detection over wearables for users before describing their CNN 
architecture at a high level. The authors summarize related work regarding using 2D CNNs for information extractions from video data and static images, and 
proposed 3D CNNs for the extraction of ‘discriminative features’ and dimensions. The methodology for this approach is explained, where feature detection is used to 
separate between ‘gesture’ and ‘no gesture’ classes, followed by the use of a 3D CNN for gesture classification and post-processing. The authors also describe 
their experiment to evaluate the performance of this model in offline results against two datasets, EgoGesture and nvGesture. The authors briefly touch on 
directions in which they would like to take their personal research.  
- [8] M. Abavisani et al., “Improving the performance of unimodal dynamic hand gesture recognition with multimodal training,” IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 1165-1174 
  - https://openaccess.thecvf.com/content_CVPR_2019/papers/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_HandGesture_Recognition_With_Multimodal_CVPR_2019_paper.pdf  
  - This work introduces a new framework which leverages the knowledge from multimodal data during training and improves the performance of a unimodal 
system during testing. In this work, the authors highlight the benefits of training a 3D CNN for each modality data for the recognition of dynamic hand gestures in 
order to derive a ‘common understanding’ for the contents of these modalities. Related work is covered in the fields of dynamic hand gesture recognition, 
including pre-existing feature detection and 3D-CNN based hand gesture recognition. The topics of transfer learning and multimodal fusion are briefly 
covered, before the authors describe their ‘one-3D-CNN-per-modality’ methodology in detail. The results of their experiments with multiple datasets and 
and implementation details using this method are described, before concluding with summary and implications/future directions for continued research.  


# Methods
The first step in our approach is to identify the hand. We use MediaPipe to extract the landmarks of the hand from an image. The landmarks of the hand are defined as follows: 

<img src="docs/assests/images/hand_diagram.jpeg?raw=true" alt="Interest Points" width="1000"/>

Each landmark includes x, y, and z coordinates for both the image space and the global position estimates.  

We can compare the coordinates of each interest point in a finger to determine the shape and position. We want to be able to calculate the angle of bend at every knuckle to determine if each finger is extended or bent. There are 4 points for each finger, so three vectors can be drawn by connecting these points. We obtain the cosine theta between vectors by dividing the inner product by the product of the magnitude of the two vectors. If the cosine theta is less than a threshold, then it means that the finger is bent. 

Velocity is currently determined using the distance of each individual landmark from its previous position, for all 21 landmarks that comprise the hand, every frame that a hand is on-screen. As the hand moves between frames, the Euclidean distance of the current coordinates and the previous set of coordinates are calculated using NumPy arrays and vectorized array operations, to determine the velocity of each landmark in terms of change of distance per frame. This metric will likely require refinement as the project continues to a more standard measurement (units per second, for example) but right now the current velocity can be used as a visualization of the extent of change between contiguous hand positions. 

Detecting hand shape can be done with a simple linear model since we have the hands interest points. We define a model with 3 layers, the input size is 63, then a hidden layer of size 256, and finally an output layer of size 2. We use ReLU activation functions between the layers and use SoftMax on the output. We had to collect data to train this model, and we used 1000 datapoints where the index finger was pointing right, and 1000 datapoints where the hand was in various shapes. 

# Experiments
For detecting straight and bent fingers, we found that our fingers will either be straight or bent at a large angle. However, the interest points from Mediapipe tend to represent the points in a straight line, rather than at the actual angle. So even when the finger is bent, the interest points still shows that the finger is straight. In the end, we use a large threshold of 0.9, so that the detector is more sensitive to the bending of the fingers. 

Our method for calculating velocity gives us a sense of a moving versus a still hand. We still need to work out how we can better use this information to inform our system of the gesture being performed. We aim to use similar experimental techniques to gauge whether our system can correctly detect and classify more sophisticated gestures and motions. 

To experiment with detecting shapes, we designed a neural network to detect whether a finger is pointing to the right. The accuracy it achieves for our dataset is 99%. Using this model with MediaPipe we can see the predictor is able to accurately predict whether the hand is pointing right. There are several limitations to this model though. One is that it is not very generic, and the prediction results are inaccurate with different users. This stems from using such a small dataset for training and will need to be addressed for the final report. Another drawback is that it is only detecting one shape, this limits our ability to communicate effectively, so we will need to improve the model or investigate other methods for detecting shape.  
<img src="docs/assests/images/bothdetect.png?raw=true" alt="Detection" width="1000"/>

## What's Next

Right now, our processes are not connected, meaning that each one performs individually and isn’t dependent on the others in any way. Hence, our next step is to look at how we can combine the information we are gathering based on shape, position, and velocity, and output something meaningful to the computer to describe motion.  
- Detection Model
    - Our model is very simple and only detects one shape. We need to broaden this to include many shapes. This may require training on a larger network and using a premade dataset.  
    - Anticipated Completion Date: 4/1  
- Hand Velocity  
    - Refine and improve our results for detecting how fast the hand in moving  
    - Anticipated Completion Date: 4/1  
- Combining Techniques  
    - We need to use all the information we are gathering from the hand to determine what motion the user is describing and translate that to the computer.  
    - Anticipated Completion Date: 4/10  
- Application  
    - Stretch goal of applying gesture detection to some real-world application or simulator.  
    - Anticipated Completion Date: 4/15    

## Work Division

| Member                       | Barnabas Borcsok | James Yu | Joseph Hardin | Justin Wit |
|:-----------------------------|:-----------------|:---------|:--------------|:-----------|
| Interesting Points Detection | V                |          |               |            |
| MediaPipe Setup              | V                |          |               |            |
| Neural Network               |                  |          |               | V          |
| Finger Extend or Bend        |                  | V        |               |            |
| Velocity Detection           |                  |          | V             |            |
| Introduction                 |                  | V        |               | V          |
| Related Works                |                  | V        | V             |            |
| Methods                      |                  | V        | V             | V          |
| Experiments                  |                  | V        | V             | V          |
| What's Next                  |                  |          |               | V          |
| Github Setup                 | V                |          |               |            |
| Github Page Setup            |                  | V        |               |            |
