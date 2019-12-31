"""
Deep Deterministic Policy Gradient agent
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

#adaptions by ckalla

# Torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from collections import deque
from matplotlib import pyplot as plt


# Lib
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import os

# Files
from thirdParty.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from thirdParty.replaybuffer import Buffer
from thirdParty.actorcritic import Actor, Critic

# Hyperparameters
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
MINIBATCH_SIZE = 128
NUM_EPISODES = 10000
SIGMA = 0.2
CHECKPOINT_DIR = './checkpoints/'
BUFFER_SIZE = 100000
DISCOUNT = 0.99
TAU = 0.001
WARMUP = 256
NETWORKUPDATEINTERVAL = 100
UPDATESTEPSPERUPDATEINTERVAL = 10

class DDPG:
    def __init__(self, env, env_info, brain_name):
        self.env = env
        self.brain_name = brain_name
        self.stateDim = env_info.vector_observations.shape[1]
        brain = env.brains[brain_name]
        self.actionDim = brain.vector_action_space_size
        self.actor = Actor(self.actionDim,self.stateDim ).cuda()
        self.critic = Critic(self.actionDim,self.stateDim).cuda()
        self.targetActor = deepcopy(Actor(self.actionDim,self.stateDim)).cuda()
        self.targetCritic = deepcopy(Critic(self.actionDim,self.stateDim)).cuda()
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        

    #soft update target network with parameter tau
    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)


    def loadExistingCheckPoint(self):
        checkpointFiles = os.listdir(self.checkpoint_dir)
        checkpointList = list(map(lambda x: int(x[2:].split(".")[0]), checkpointFiles))
        checkpointList.sort()
        largestCheckpoint = checkpointList[-1]
        self.loadCheckpoint(os.path.join(self.checkpoint_dir, "ep{0}.pth.tar".format(largestCheckpoint)))

    def runTrainedAgent(self,nSteps=10000):
        self.loadExistingCheckPoint()
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        currentState = env_info.vector_observations[0]

        for i in range(nSteps):
            self.actor.eval()
            # action = self.getMaxAction(currentState)
            with torch.no_grad():
                currentState_expanded = np.expand_dims(currentState, axis=0)
                currentStateTensor = Variable(torch.FloatTensor(currentState_expanded), volatile=True).cuda()
                action = self.actor(currentStateTensor)
                action = torch.clamp(action, min=-1, max=1)
            action_numpy = action.data.cpu().numpy()
            env_info = self.env.step(action_numpy)[self.brain_name]
            next_state = env_info.vector_observations[0]
            currentState = next_state

    def train(self, resumeTraining=False):
        print('Training started...')
        if resumeTraining:
            print("Resuming training")
            self.loadExistingCheckPoint()

        epochRewards = deque(maxlen=100)

        for i in range(self.start, self.end):
            print("Episode:{0}".format(i))
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            currentState = env_info.vector_observations[0]
            done = False
            ep_reward = 0
            actionsTaken = 0
            while not done:
                self.actor.eval()
                #action = self.getMaxAction(currentState)
                with torch.no_grad():
                    noise = Variable(torch.FloatTensor(self.noise()), volatile=True).cuda()
                    currentState_expanded = np.expand_dims(currentState,axis=0)
                    currentStateTensor = Variable(torch.FloatTensor(currentState_expanded), volatile=True).cuda()
                    action =  self.actor(currentStateTensor)
                    action = action + noise
                    action = torch.clamp(action, min=-1, max=1)
                self.actor.train()
                action_numpy = action.data.cpu().numpy()
                #Step episode
                env_info = self.env.step(action_numpy)[self.brain_name]  # send all actions to tne environment
                actionsTaken += 1
                next_state = env_info.vector_observations[0]  # get next state (for each agent)
                reward = env_info.rewards[0]  # get reward (for each agent)
                done = env_info.local_done[0]  # see if episode finished
                ep_reward += reward  # update the score (for each agent)


                # Update replay bufer
                self.replayBuffer.append((currentState, action_numpy, next_state, reward, done))
                currentState = next_state  # roll over states to next time step
                
                # Training loop
                #check if replay buffer contains enough samples
                if len(self.replayBuffer) >= self.warmup:
                    #do backward prop only every NETWORKUPDATEINTERVAL steps
                    if actionsTaken % NETWORKUPDATEINTERVAL == 0:
                        #do UPDATESTEPSPERUPDATEINTERVAL updates
                        for nStep in range(UPDATESTEPSPERUPDATEINTERVAL):
                            #sample batch from replay memory
                            curStateBatch, actionBatch, nextStateBatch, rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                            qPredBatch = self.critic(curStateBatch, actionBatch)
                            #use target actor network to get next action
                            actions_next = self.targetActor(nextStateBatch)
                            #use target critic network to get q value for next (state,action) tuple
                            q_targets_next = self.targetCritic(nextStateBatch, actions_next)
                            #compute the groundtruth q value by summing the current return and the discounted future return (bellman equation)
                            qTargetBatch = rewardBatch + (DISCOUNT * q_targets_next * (1 - terminalBatch))

                        # Critic update
                            self.criticOptim.zero_grad()
                            criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                            #print('Critic Loss: {}'.format(criticLoss))
                            criticLoss.backward()
                            #gradient clipping ensures that we do not encounter strange behaviour in later training stages
                            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)

                            self.criticOptim.step()

                        # Actor update
                            self.actorOptim.zero_grad()
                            #the actor loss is the negative value estimate of the critic given the current state and the action estimated by the actor network
                            actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                            #print('Actor Loss: {}'. format(actorLoss))
                            actorLoss.backward()
                            self.actorOptim.step()

                        # Update Targets
                            self.updateTargets(self.targetActor, self.actor)
                            self.updateTargets(self.targetCritic, self.critic)
            print("Ep reward: {0}".format(ep_reward))
            epochRewards.append(ep_reward)
            if len(epochRewards) >= 100:
                averageReward = sum(epochRewards)/len(epochRewards)
                #environment is solved if the average score over the last 100 epochs exceeds 30
                if averageReward > 30.0:
                    self.save_checkpoint(i)
                    print("Task solved within {0} epochs!".format(i))
                    break
            #save checkpoints every 100 iterations
            if i % 100 == 0:
                self.save_checkpoint(i)
            self.rewardgraph.append(ep_reward)


    def save_checkpoint(self, episode_num):
        checkpointName = self.checkpoint_dir + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'replayBuffer': self.replayBuffer,
            'rewardgraph': self.rewardgraph
            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOpt'])
            self.criticOptim.load_state_dict(checkpoint['criticOpt'])
            self.replayBuffer = checkpoint['replayBuffer']
            self.rewardgraph = checkpoint['rewardgraph']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')

    def plotRewardGraph(self):
        xlabel = range(0,len(self.rewardgraph))
        plt.plot(xlabel,self.rewardgraph)
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.savefig("rewardGraph.png")
        plt.imshow()


