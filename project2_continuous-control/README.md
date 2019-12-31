# Project details
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
## State space
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
## Action space
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
## Solving the environment
The environment is considered as solved if the agent is possible to get an average reward of +30 within 100 episodes
## Getting started
* Install the Reacher single agent environment environment provided by Udacity (i.e. move the binaries and the Banana_Data folder into the source folder)
* Install pytorch
* Install matplotlib for result visualization
# Instructions
* run train.py to start training. After training has finished the trained model checkpoints are stored in the checkpoints folder, the plot is shown and stored as rewardGraph.png.
Furthermore a simulation of the agent is run in test mode using the best actor net.