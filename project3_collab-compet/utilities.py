import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt

#this module contains utility methods (e.g. for plotting the reward graph or computing the moving average)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plotRewardGraph(plotDir,inputRewards,solveThreshold):
    plt.title("Reward plot for Udacity RL Tennis Environment")
    agent1Rewards = [element[0] for element in inputRewards]
    agent1Rewards_ma = moving_average(agent1Rewards,100)
    agent2Rewards = [element[1] for element in inputRewards]
    agent2Rewards_ma = moving_average(agent2Rewards, 100)
    maxRewards = [np.max(element) for element in inputRewards]
    maxRewards_ma = moving_average(maxRewards, 100)
    plt.plot(range(len(agent1Rewards_ma)), agent1Rewards_ma,color='r',label='Agent 1')
    plt.plot(range(len(agent2Rewards_ma)), agent2Rewards_ma, color='b', label='Agent 2')
    plt.plot(range(len(maxRewards_ma)), maxRewards_ma, color='g', label='Max')
    plt.xlabel("nEpisode")
    plt.ylabel("Rewards (running average over 100 episodes)")
    plt.legend(loc="center left")
    plt.axhline(linewidth=2, color='r', y=solveThreshold)
    plt.savefig(os.path.join(plotDir,"scoresPerEpisode.png"))
    plt.show()
