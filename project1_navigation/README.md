# Project details
The environment consists of yellow and green bananas. The goal is to program and agent that collects as much yellow bananas as possible while avoiding to collect blue bananas.
Yellow bananas give a reward of +1 while blue ones give a negative reward of -1.
## State space
The state space is a vector with 37 dimensions, representing perceptual information of the environment the agent is currently observing. This is a simplification, the agent could
be trained with image input as well. However, the neural network would be a bit more complicated then, containing convolution layers at the beginning.
## Action space
The agent has 4 different actions (0=forward,1=backward,2= turn left,3= turn right) available.
## Solving the environment
The environment is considered as solved if the agent is possible to get an average reward of +13 per episode
## Getting started
* Install the Unity Banana environment provided by Udacity (i.e. move the binaries and the Banana_Data folder into the source folder)
* Install pytorch
* Install matplotlib for result visualization
* (Optional) Install keras in order to run the experimental Keras implementation navigation_keras.py
# Instructions
* run navigation_pytorch.py to start training. After training has finished the trained model is stored in the models folder, the plot is shown and stored in the plots folder
* to run in evaluation mode set the variable **trainMode** to **False** at the beginning of the script
* (Optional) For training with keras run navigation_keras.py. This file is not documented that well. I attached it as proof of concept.