"""
Definitions for Actor and Critic
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

HID_LAYER1_DIM = 256
HID_LAYER2_DIM = 128
WFINAL = 0.003



class Actor(nn.Module):
    """Defines actor network"""
    def __init__(self, actionDim,stateDim):
        super(Actor, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        
        self.norm0 = nn.BatchNorm1d(self.stateDim)
                                    
        self.fc1 = nn.Linear(self.stateDim, HID_LAYER1_DIM)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())            
        self.bn1 = nn.BatchNorm1d(HID_LAYER1_DIM)
                                    
        self.fc2 = nn.Linear(HID_LAYER1_DIM, HID_LAYER2_DIM)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
                                    
        self.bn2 = nn.BatchNorm1d(HID_LAYER2_DIM)
                                    
        self.fc3 = nn.Linear(HID_LAYER2_DIM, self.actionDim)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
            
    def forward(self, ip):
        ip_norm = self.norm0(ip)                            
        h1 = self.ReLU(self.fc1(ip_norm))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(h1_norm))
        h2_norm = self.bn2(h2)
        action = self.Tanh((self.fc3(h2_norm)))
        return action
        

class Critic(nn.Module):
    """Defines critic network"""
    def __init__(self, actionDim,stateDim):
        super(Critic, self).__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        
        self.fc1 = nn.Linear(self.stateDim, HID_LAYER1_DIM)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.bn1 = nn.BatchNorm1d(HID_LAYER1_DIM)
        self.fc2 = nn.Linear(HID_LAYER1_DIM + self.actionDim, HID_LAYER2_DIM)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        
        self.fc3 = nn.Linear(HID_LAYER2_DIM, 1)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU()
        
    def forward(self, state, action):
        h1 = self.ReLU(self.fc1(state))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(torch.cat([h1_norm, action], dim=1)))
        Qval = self.fc3(h2)
        return Qval
        

