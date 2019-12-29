# Project Report
## Learning Algorithm
In order to solve this continuous control task the DPPG (deep deterministic policy gradient) algorithm is applied. It is suitable in this case as both state and action space are continuous and the environment is episodic.
The algorithm can be classified as actor critic method, i.e. it makes use of policy and value function optimization at the same time. The main components of the algorithm are the actor and the critic neural network.
The actor network gets the state as input and outputs the action. The critic gets the state and action as input and outputs the approximated value for the given (state,action) tuple.
In every iteration the critic loss is computed by using the squared difference between the true reward (observing the environment) and the computed current and discounted reward using the critic network (Bellman equation).
The actor loss in each iteration is computed by the negative value output of the critic network given the current state and the action computed by the current actor network as input.
As in the Deep Q-Learning algorithm local and target networks are used (both for actor and critic) to make gradient updates more stable and facilitate convergence.
I applied slight modifications of the algorithm in my implementation:
* Gradient updates are not done in every iteration, but only when sufficient number of samples have been added (NETWORKUPDATEINTERVAL)
* During each update UPDATESTEPSPERUPDATEINTERVAL are performed on randomly sampled batches from the replay memory


The algorithm is applicable here, because the state space is continuous, we have a discrete set of actions and obtain a reward and the next state from the environment after taking an action.
In every iteration we sample a set of (state,action,next_state, reward) tuples which we add to our
replay memory. The selection of actions is done in an epsilon greedy way, i.e. that we choose a random action with probability epsilon and with probability 1-epsilon we select the action which gives
us the maximum expected reward using the current Q network.

In the current implementation a queue with the capacity of 10000 is used for the replay memory storage, but I also experimented with an implementation of prioritized experience replay. The idea
here is to prioritize the samples leading to a larger error value with respect to the current Q network prediction, as it is assumed that learning is faster because of higher information
content of these samples.

The gradient update is based on the mean squared error loss (i.e. the squared difference between the predicted reward of the network and the "ground truth" reward obtained from a random batch of sampled observations from the replay memory).
In order to make the training more stable the algorithm makes use of two neural networks (prediction and target network) with identical architecture. The prediction network is updated in every iteration while the target network (used for computing the expected reward
of the next state from the environment observation tuple) is kept fixed for a number of iterations. After this number of steps the weights are transferred. The whole process makes the training more stable, such that
faster convergence is reached. Weight update can also be achieved with a soft update using parameter tau which is used in this implementation.
I found out that the size of the replay memory buffer, epsilon and epsilon decay and the decision if hard or soft updates are used has a crucial impact on the overall algorithm performance.

The network structure is quite simple, perhaps it could be optimized by more sophisticated parameter tuning. It looks as follow:
Critic network (Input: (state,action), Output: value)

* Dense layer - input: 33 (state size) output: 256
* Dense layer - input: 256 output 128
* Dense layer - input: 128 output: 1

Actor network (Input: state, Output: action)

* Dense layer - input: 33 (state size) output: 256
* Dense layer - input: 256 output 128
* Dense layer - input: 128 output: 4

As activation functions ReLU is used which is the standard activation function for most deep learning tasks. As the state space has only 37 dimension it is sufficient to only use
fully connected layers. If the input would consist of RGB images we would need additional convolution layers.

## Hyperparameters used
* Start value of epsilon: 1.0 (the start value has to be high as the task demands a high extent of exploration of the environment)
* End value of epsilon:  0.01 (we still want to select random actions and ever stop exploration of the environment)
* Epsilon decay: 0.999 (the decay factor has to be high, as we want to keep a high extent of exploration over the episodes. Otherwise we could get stuck in local minima)
* Size of hidden layers of the neural net: 256-128 (this was the first shot and works well)
* tau (for weight transfer of local and target network) : 0.001
* Batch Size: 64 (could also be 32, makes not much difference for this task I guess)
* Discount Factor: 0.99
* Learning rate of Adam solver: 0.001 (works best for a lot of machine learning tasks, also in this case)

## Results
 ![]( plots/scoresPerEpisode.png)

## Future work
* Implement prioritized experience replay (per.py already contains a github implementation, but for this environment it did not speed up training, or I made a mistake when using it...)
* Find out why the Keras implementation converges slower than the pytorch one
* Learn directly from the input images and not from the low dim observation state vector
