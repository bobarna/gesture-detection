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
