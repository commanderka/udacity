# individual network settings for each actor + critic pair
# see networkforall for details

from Networks import Critic,Actor
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np
from copy import deepcopy

# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, stateSize,actionSize, lr_actor=1.0e-3, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()


        #initialize actor, target actor, critic and target critic for each agent
        self.actor = Actor(actionSize,stateSize).to(device)
        self.critic = Critic(actionSize,stateSize).to(device)
        self.target_actor = deepcopy(Actor(actionSize,stateSize)).to(device)
        self.target_critic = deepcopy(Critic(actionSize,stateSize)).to(device)

        #initialize OU noise
        self.noise = OUNoise(mu=np.zeros(actionSize), sigma=0.01)

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        #we choose adam as an optimizer for both actor and critic
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)


    #compute action of the local actor by the current actor and add OU noise on top
    def act(self, obs, noise=0.0):
        self.actor.eval()
        action = self.actor(obs) + noise*torch.FloatTensor(self.noise()).to(device)
        self.actor.train()
        return action

    #compute action of the target actor by the current actor and add OU noise on top
    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise*torch.FloatTensor(self.noise()).to(device)
        return action
