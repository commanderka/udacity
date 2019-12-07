import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optimization

from collections import deque

#class representing the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers = [256,128]):
        super(NeuralNetwork, self).__init__()

        #the network consists of an input and output layer and 2 hidden layers with ReLU activations
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layers[1], action_size)

    def forward(self,inputState):
        state = self.fc1(inputState)
        state = self.relu1(state)
        state = self.fc2(state)
        state = self.relu2(state)
        qValues = self.fc3(state)
        return qValues

#class representing the agent
class MyTorchAgent:
    def __init__(self, state_size, action_size,tau, batchSize,gamma):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batchSize = batchSize
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        self.qnetwork_local = NeuralNetwork(state_size, action_size).to( self.device)
        self.qnetwork_target = NeuralNetwork(state_size, action_size).to( self.device)
        self.optimizer = optimization.Adam(self.qnetwork_local.parameters(), lr=1e-3)
        self.updateInterval = 10
        self.hardUpdateInterval = 10
        self.optimizationStepsTaken = 0

        #Replay memory
        self.memory = deque(maxlen=10000)
        #counter variable to count the steps taken until update_interval is reached
        self.nSteps = 0

    def sample(self):
        #Randomly sample a batch of observations from memory
        observations = random.sample(self.memory, k=self.batchSize)

        states = torch.from_numpy(np.vstack([e[0] for e in observations if e is not None])).float().to( self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in observations if e is not None])).long().to( self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in observations if e is not None])).float().to( self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in observations if e is not None])).float().to( self.device)
        doneValues = torch.from_numpy(np.vstack([e[4] for e in observations if e is not None]).astype(np.uint8)).float().to( self.device)

        return (states, actions, rewards, next_states, doneValues)

    def __add(self, state, action, reward, next_state, done):
        #Add a new experience to memory.
        e = (state, action, reward, next_state, done)
        self.memory.append(e)

    def __hard_update(self,local_model,target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def __soft_update(self, local_model, target_model, tau):
        '''
        Do a soft update on the model parameters.
        tau_target = tau*local_weights + (1 - tau)*target_weights

            local_model (PyTorch model): original pytorch model
            target_model (PyTorch model): target model to update
            tau (float): parameter defining the extent of the update

        '''
        for target_parameters, local_parameters in zip(target_model.parameters(), local_model.parameters()):
            target_parameters.data.copy_(tau * local_parameters.data + (1.0 - tau) * target_parameters.data)

    def __learnFromExamples(self, observations, gamma):
        '''
        Update the value parameters using a given batch of observations.

            observations : tuple with form (state, action, reward, new state, done)
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, doneValues = observations

        localNetworkPredictions = self.qnetwork_local.forward(states)
        #we care only about the network predictions at the taken actions (i.e. we do not consider all predicted action values)
        localNetworkPredictions_atTakenActions = localNetworkPredictions[torch.arange(self.batchSize, dtype=torch.long), actions.reshape(self.batchSize)]
        localNetworkPredictions_atTakenActions = localNetworkPredictions_atTakenActions.reshape((self.batchSize, 1))

        # using the greedy policy (q-learning)
        targetNetworkPredictions = self.qnetwork_target.forward(next_states)
        #select action which maximizes the reward acc. to algorithm
        _, targetNetworkPredictions_maxActions = torch.max(targetNetworkPredictions,dim=1)
        #get the prediction value of the maximizing action
        targetNetworkPredictions_atMaxAction = targetNetworkPredictions[torch.arange(self.batchSize, dtype=torch.long), targetNetworkPredictions_maxActions.reshape(self.batchSize)]

        #if the state should be final state (done=True)_action_values_target is zero
        state_action_values_target = targetNetworkPredictions_atMaxAction * (1 - doneValues.reshape(self.batchSize))
        state_action_values_target = state_action_values_target.reshape((self.batchSize, 1))
        trueValues = rewards + gamma * state_action_values_target

        #we use mean squared error loss here (the loss is the difference between the local network prediction and the reward received by environment interaction + discounted next state prediction of the target network)
        loss = F.mse_loss(localNetworkPredictions_atTakenActions,trueValues)
        # clear gradients
        self.optimizer.zero_grad()
        #do backpropagation of the loss
        loss.backward()
        self.optimizer.step()
        self.optimizationStepsTaken += 1

        # update the network with soft updates using the parameter tau
        self.__soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # alternatively update the weights of the target network with the weights of the local network
        '''
        if self.optimizationStepsTaken % self.hardUpdateInterval == 0:
            self.optimizationStepsTaken = 0
            self.__hard_update(self.qnetwork_local, self.qnetwork_target)
        '''

    #method to return an action for a given state (according to the current prediction of the local network)
    def doAction(self, state, epsilon=0, trainMode=True):
        '''
        Returns actions for given state according to current policy
            state (array_like): current state
            epsilon (float): epsilon for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        if trainMode:
            self.qnetwork_local.train()

        # select action with epsilon greedy method
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size))

    #method for gathering observations from the environment/ doing update steps
    def doStep(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.__add(state, action, reward, next_state, done)
        self.nSteps = (self.nSteps + 1) % self.updateInterval
        # only do learning every updateInterval steps
        if self.nSteps == 0:
            # check if enough samples are available in replay memory and learn
            if len(self.memory) > self.batchSize:
                observations = self.sample()
                self.__learnFromExamples(observations, self.gamma)


