# Project details
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

## State space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
be trained with image input as well. However, the neural network would be a bit more complicated then, containing convolution layers at the beginning.
## Action space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
## Solving the environment
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.
* The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
## Getting started
* Install the Unity Tennis environment(i.e. move the binaries and the Tennis_Data folder into the source folder)
* Install pytorch
* Install matplotlib for result visualization
# Instructions
* run main.py to start training. After training has finished the trained model is stored in the model_dir folder, the plot is shown and stored in the plots folder
* to run in evaluation mode set the variable **trainMode** to **False** at the beginning of the script. A simulation with the best trained agents is run in this case.