# Project details

## State space
The state space is a vector with 33 dimensions, representing perceptual information of the environment the agent is currently observing. This is a simplification, the agent could
be trained with image input as well. However, the neural network would be a bit more complicated then, containing convolution layers at the beginning.
## Action space
The action space is a continuous space with dimensionality 4
## Solving the environment
The environment is considered as solved if the agent is possible to get an average reward of +30 within 100 episodes
## Getting started
* Install the Reacher single agent environment environment provided by Udacity (i.e. move the binaries and the Banana_Data folder into the source folder)
* Install pytorch
* Install matplotlib for result visualization
# Instructions
* run navigation_pytorch.py to start training. After training has finished the trained model is stored in the models folder, the plot is shown and stored in the plots folder
* to run in evaluation mode set the variable **trainMode** to **False** at the beginning of the script
* (Optional) For training with keras run navigation_keras.py. This file is not documented that well. I attached it as proof of concept.