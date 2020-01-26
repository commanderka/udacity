# main function that sets up environments
# perform training loop

from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
import os
from utilities import transpose_list, transpose_to_tensor, plotRewardGraph
from unityagents import UnityEnvironment
from collections import deque
from matplotlib import pyplot as plt



def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 100000
    batchsize = 256
    # how many episodes to save policy and gif
    save_interval = 1000
    #specifies if training is done or a simulation of the trained agents
    trainMode = True
    #set noise to zero in train mode, otherwise we multiple the OU nouise with a factor of 8
    if not trainMode:
        noiseFactor = 0
    else:
        # amplitude of OU noise
        # this slowly decreases to 0
        noiseFactor = 8

    #factor by which the noise is reduced in each episode (the noise shall be large at the beginning and decrease as training progresses)
    noise_reduction = 0.99
    #defines the update frequency in episodes of the used neural networks
    updateInterval = 1
    #defines how many update steps are done per episode
    nUpdatesPerUpdateInterval = 2
    #interval used to save the models
    saveInterval = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #list storing all agent scores
    allScores = []
    #threshold to reach to consider the environment as solved
    #we track the sum of rewards of each agent for each episode, take the max value of this and average this value over 100 episodes
    #this means also that one agent can be better than the other, as the max score is taken
    solveThreshold = 0.5

    model_dir= os.getcwd()+"/model_dir"
    plot_dir = os.getcwd()+"/plots"
    os.makedirs(model_dir, exist_ok=True)


    # keep 100000 episodes worth of replay
    buffer = ReplayBuffer(100000)
    
    # initialize multi aggent ddpg object (containing two DDPG agents)
    maddpg = MADDPG()

    #load best trained agents in test mode
    if not trainMode:
        print("Loading best pretrained weights...")
        maddpg.loadModel(os.path.join(model_dir,"episode-2000.pt"))

    #init environment
    env = UnityEnvironment(file_name="Tennis.exe")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=trainMode)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    scores = np.zeros(num_agents)

    # training loop

    for episode in range(0, number_of_episodes):
        noiseFactor = noiseFactor * noise_reduction
        #print("Episode number: {0}".format(episode))


        #we sample a number of transitions before the updates are done
        #this ensures having a meaningful replay buffer such that update steps make sense
        nSamplesToAdd = batchsize
        nSamplesAdded = 0
        # save info or not
        nStep = 0
        env_info = env.reset(train_mode=trainMode)[brain_name]
        currentObservation = env_info.vector_observations
        scoresPerEpisode = np.zeros(num_agents)
        while (True):
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed

            currentObservation_torch = torch.from_numpy(currentObservation).float().to(device)
            observationList = []
            observationList.append(currentObservation_torch[0].unsqueeze(0))
            observationList.append(currentObservation_torch[1].unsqueeze(0))

            actions = maddpg.act(observationList, noise=noiseFactor)

            action1 = np.clip(actions[0].detach().cpu().numpy(),-1,1)
            action2 = np.clip(actions[1].detach().cpu().numpy(),-1, 1)
            actions_numpy = np.vstack((action1,action2))

            env_info = env.step(actions_numpy)[brain_name]
            rewards = env_info.rewards
            scoresPerEpisode += rewards
            dones = env_info.local_done
            # add data to buffer
            next_observation = env_info.vector_observations
            transition = (currentObservation, actions_numpy, next_observation, rewards, dones)
            buffer.push(transition)
            nSamplesAdded +=1
            currentObservation = next_observation
            if np.any(dones):
                #episode is over, so append agent scores to overall scores and reset the counter of scores accumulated for the episode
                allScores.append(scoresPerEpisode)
                scoresPerEpisode = np.zeros(num_agents)
                #break the loop if we have gathered enough samples
                if nSamplesAdded > nSamplesToAdd:
                    break
                env_info = env.reset(train_mode=trainMode)[brain_name]
                currentObservation = env_info.vector_observations



        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % updateInterval == 0:
            for nUpdate in range(nUpdatesPerUpdateInterval):
                for a_i in range(num_agents):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i)
                maddpg.update_targets() #soft update the target network towards the actual networks
        maxRewards = [np.max(element) for element in allScores]
        meanScoreOver100Episodes = np.mean(maxRewards[-100:])
        if meanScoreOver100Episodes > solveThreshold and trainMode:
            print("Environment solved in {0} episodes!".format(episode) )
            maddpg.saveModel(model_dir, episode)
            plotRewardGraph(plot_dir,allScores, solveThreshold)
            break

        if (episode+1) % 10 == 0 or episode == number_of_episodes-1:
            print("Episode: {0}".format(episode))
            print("Mean score over 100 episodes: {0}".format(meanScoreOver100Episodes))
            agent0_reward = []
            agent1_reward = []

        if episode % saveInterval == 0 and trainMode:
            maddpg.saveModel(model_dir,episode)


if __name__=='__main__':
    main()
