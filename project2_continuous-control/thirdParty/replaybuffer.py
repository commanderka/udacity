"""
Replay Buffer 
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

import random
import numpy as np
import torch
from collections import deque
MINIBATCH_SIZE = 64

class Buffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen=self.limit)
        
    def __len__(self):
        return len(self.data)
    
    def sample_batch(self, batchSize):
        if len(self.data) < batchSize:
            warnings.warn('Not enough entries to sample without replacement.')
            return None
        else:

            batch = random.sample(self.data, batchSize)
            curState = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float().cuda()
            action = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).float().cuda()
            nextState = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float().cuda()
            reward = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float().cuda()
            terminal = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None]).astype(np.uint8)).float().cuda()
        return curState, action, nextState, reward, terminal
                  
    def append(self, element):
        self.data.append(element)  
