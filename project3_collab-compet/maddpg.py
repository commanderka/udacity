# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import torch.nn.functional as F
import os

class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.1):
        super(MADDPG, self).__init__()


        # critic input = obs_full + actions = 2*24+2*2=52
        self.maddpg_agent = [DDPGAgent(24, 2),
                             DDPGAgent(24, 2),
                             ]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.nAgents = len(self.maddpg_agent)

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def transpose_to_tensor(input_list):
        make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
        return list(map(make_tensor, zip(*input_list)))

    #update function for a specific agent
    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        #batch of observations of the first agent
        obs_agent1 = torch.from_numpy(np.vstack([e[0][0] for e in samples if e is not None])).float().to(device)
        # batch of observations of the second agent
        obs_agent2 =  torch.from_numpy(np.vstack([e[0][1] for e in samples if e is not None])).float().to(device)
        #list containing both agents observations
        obs = [obs_agent1, obs_agent2]
        #concatenation of both agents observations
        obs_full = torch.from_numpy(np.vstack([np.concatenate((e[0][0],e[0][1])) for e in samples if e is not None])).float().to(device)
        # concatenation of both agents actions
        action_full = torch.from_numpy(np.vstack([np.concatenate((e[1][0],e[1][1])) for e in samples if e is not None])).float().to(device)
        #batch of the next observations of agent 1
        next_obs_agent1 = torch.from_numpy(np.vstack([e[2][0] for e in samples if e is not None])).float().to(device)
        # batch of the next observations of agent 2
        next_obs_agent2 = torch.from_numpy(np.vstack([e[2][1] for e in samples if e is not None])).float().to(device)
        # list containing both agents next observations
        next_obs = [next_obs_agent1,next_obs_agent2]
        # concatenation of both agents next observations
        next_obs_full = torch.from_numpy(np.vstack([np.concatenate((e[2][0], e[2][1])) for e in samples if e is not None])).float().to(device)

        reward = torch.from_numpy(np.vstack([e[3] for e in samples if e is not None])).float().to(device)
        done = torch.from_numpy(np.vstack([e[4] for e in samples if e is not None]).astype(np.uint8)).float().to(device)


        agent = self.maddpg_agent[agent_number]
        #compute the next actions of all agents using the target networks (this is used as input for the critic)
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        #compute the q value of the next observation given the previously computed target actions
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full,target_actions)

        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # we have to care about the correct agent number here
        y = reward[:,agent_number].view(-1,1) + self.discount_factor * q_next * (1 - done[:,agent_number].view(-1,1))
        #compute the q value of the given state/action pair
        q = agent.critic(obs_full,action_full)

        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        critic_loss = F.mse_loss(q,y)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        # optimize the critic by backpropagation
        agent.critic_optimizer.step()


        #update actor network using policy gradient

        q_input = [ self.maddpg_agent[i].actor(ob) for i, ob in enumerate(obs) ]
        # combine all the actions and observations for input to critic
        q_input = torch.cat(q_input, dim=1)

        agent.actor_optimizer.zero_grad()
        # get the policy gradient
        # the actor is good if the value of the returned action evaluated by the critic is high, so the negative of it has to be low
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        # optimize the actor by backpropagation
        agent.actor_optimizer.step()


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)


    #function to save the model parameters of all agents
    def saveModel(self, modelDir, nEpisode):
        # saving model
        save_dict_list = []

        for i in range(2):
            save_dict = {'actor_params': self.maddpg_agent[i].actor.state_dict(),
                         'actor_optim_params': self.maddpg_agent[i].actor_optimizer.state_dict(),
                         'critic_params': self.maddpg_agent[i].critic.state_dict(),
                         'critic_optim_params': self.maddpg_agent[i].critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)

            torch.save(save_dict_list, os.path.join(modelDir, 'episode-{}.pt'.format(nEpisode)))

    #function to load the model parameters of all agents
    def loadModel(self, modelPath):
        state_dict_list = torch.load(modelPath)
        for i in range(2):
            self.maddpg_agent[i].actor.load_state_dict(state_dict_list[i]['actor_params'])
            self.maddpg_agent[i].actor_optimizer.load_state_dict(state_dict_list[i]['actor_optim_params'])
            self.maddpg_agent[i].critic.load_state_dict(state_dict_list[i]['critic_params'])
            self.maddpg_agent[i].critic_optimizer.load_state_dict(state_dict_list[i]['critic_optim_params'])
            
            
            




