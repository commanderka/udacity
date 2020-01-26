# Project Report
## Learning Algorithm

![]( maddpg_pseudocode.png)
![]( multi-agent-actor-critic.png)


The network structure is quite simple, perhaps it could be optimized by more sophisticated parameter tuning. It looks as follow:

* Dense layer - input: 37 (state size) output: 256
* Dense layer - input: 256 output 128
* Dense layer - input: 128 output: (action size)

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
