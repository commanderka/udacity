
from thirdParty import ddpg
from unityagents import UnityEnvironment
import numpy as np

if __name__=='__main__':


    # select this option to load version 1 (with a single agent) of the environment
    #env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
    env = UnityEnvironment(file_name='Reacher.exe')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
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

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    ddpgAgent = ddpg.DDPG(env,env_info,brain_name)
    ddpgAgent.train()
    ddpgAgent.plotRewardGraph()
    ddpgAgent.runTrainedAgent()

    env.close()

