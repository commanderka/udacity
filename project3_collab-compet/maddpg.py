# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
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

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        obs_agent1 = torch.from_numpy(np.vstack([e[0][0] for e in samples if e is not None])).float().to(device)
        obs_agent2 =  torch.from_numpy(np.vstack([e[0][1] for e in samples if e is not None])).float().to(device)
        obs = [obs_agent1, obs_agent2]
        obs_full = torch.from_numpy(np.vstack([np.concatenate((e[0][0],e[0][1])) for e in samples if e is not None])).float().to(device)
        action_full = torch.from_numpy(np.vstack([np.concatenate((e[1][0],e[1][1])) for e in samples if e is not None])).float().to(device)
        next_obs_agent1 = torch.from_numpy(np.vstack([e[2][0] for e in samples if e is not None])).float().to(device)
        next_obs_agent2 = torch.from_numpy(np.vstack([e[2][1] for e in samples if e is not None])).float().to(device)
        next_obs = [next_obs_agent1,next_obs_agent2]
        next_obs_full = torch.from_numpy(np.vstack([np.concatenate((e[2][0], e[2][1])) for e in samples if e is not None])).float().to(device)
        reward = torch.from_numpy(np.vstack([e[3] for e in samples if e is not None])).float().to(device)
        done = torch.from_numpy(np.vstack([e[4] for e in samples if e is not None]).astype(np.uint8)).float().to(device)


        #obs_full = torch.stack(obs_full)
        #next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]



        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        #target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full,target_actions)
        
        y = reward[:,agent_number].view(-1,1) + self.discount_factor * q_next * (1 - done[:,agent_number].view(-1,1))
        #action = torch.cat(action, dim=1)
        #critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = agent.critic(obs_full,action_full)
        agent.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(q,y)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()


        #update actor network using policy gradient

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        #q_input2 = torch.cat((obs_full, q_input), dim=1)

        agent.actor_optimizer.zero_grad()
        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

        #al = actor_loss.cpu().detach().item()
        #cl = critic_loss.cpu().detach().item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

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

    def loadModel(self, modelPath):
        state_dict_list = torch.load(modelPath)
        for i in range(2):
            self.maddpg_agent[i].actor.load_state_dict(state_dict_list[i]['actor_params'])
            self.maddpg_agent[i].actor_optimizer.load_state_dict(state_dict_list[i]['actor_optim_params'])
            self.maddpg_agent[i].critic.load_state_dict(state_dict_list[i]['critic_params'])
            self.maddpg_agent[i].critic_optimizer.load_state_dict(state_dict_list[i]['critic_optim_params'])
            
            
            




