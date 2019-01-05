
# Single Agent MADDPG ???

import numpy as np
from agents.ddpg import DDPGAgent
import torch
from utils.utilities import soft_update, batch_to_tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

LOG_EVERY = 200

class SADDPG:
    def __init__(self, partial_obs_size, action_size, full_obs_size,
                 discount_factor=0.995, tau=0.01, logger = None):
        super(SADDPG, self).__init__()

        self.logger = logger
        self.action_size = action_size
        self.partial_obs_size = partial_obs_size
        self.full_obs_size = full_obs_size
        self.saddpg_agent = DDPGAgent(in_actor = partial_obs_size,
                                      out_actor = action_size,
                                      in_critic= full_obs_size + action_size,
                                      lr_actor=1.0e-2, lr_critic=1.0e-2)
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def reset(self):
        self.saddpg_agent.reset()

    def get_actors(self):
        return self.saddpg_agent.actor

    def get_target_actors(self):
        return self.saddpg_agent.target_actor

    def act(self, obs, noise=None):
        """get actions from all agents in the MADDPG object"""
        a = self.saddpg_agent.act(obs, noise)
        return a

    def target_act(self, obs):
        a = self.saddpg_agent.target_act(obs)
        return a


    def learn(self, samples):
        obs = []
        full_obs = []
        actions = []
        reward = []
        next_obs = []
        next_full_obs = []
        done = []

        for e in samples:
            obs.append(e[0])
            full_obs.append(e[1])
            actions.append(e[2])
            reward.append(e[3])
            next_obs.append(e[4])
            next_full_obs.append(e[5])
            done.append(e[6])

        obs = torch.tensor(obs, dtype=torch.float32)
        full_obs = torch.tensor(full_obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_full_obs = torch.tensor(next_full_obs, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        if self.logger is not None:
            if (self.iter + 1) % LOG_EVERY == 0:
                # Log samples
                self.log_histogram(reward, 'rewards', self.iter)
                self.log_histogram(actions, 'actions', self.iter)


        agent = self.saddpg_agent
        # To normalize or not ... ??
        # reward = (reward - reward.mean())/(reward.std() + 1e-8)

        # -------------- Critic Update ---------------- #
        agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            next_action_target = agent.target_actor(next_obs)
            _critic_input = torch.cat([next_full_obs, next_action_target], dim=1)
            next_value_target = agent.target_critic(_critic_input)
            target_value = reward + self.discount_factor*next_value_target*(1 - done)

        expected_value = agent.critic(torch.cat([full_obs, actions], dim=1)) # What we've learned so far from the data
        # huber_loss = torch.nn.SmoothL1Loss()
        l2_loss = torch.nn.MSELoss()
        critic_loss = l2_loss(expected_value, target_value.detach())
        critic_loss.backward()  # Compute the gradients wrt behavioral Q parameters
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5) # Clip dem gradients
        agent.critic_optimizer.step() # Apply gradients

        if self.logger is not None:
            if (self.iter + 1) % LOG_EVERY == 0:
                # Critic gradients
                _grads = self.get_param_grads(agent.critic.parameters())
                _norm = self.compute_param_grad_norm(agent.critic.parameters(), norm_type=2)
                self.log_scalar(_norm, 'critic_grad_norm', self.iter)


        # -------------- Actor Update ----------------- #
        agent.actor_optimizer.zero_grad() # Clear out the gradients
        guess_action = agent.actor(obs) # behavioral action.
        J = -agent.critic(torch.cat([full_obs, guess_action], dim=1)).mean()
        J.backward() # Gradients wrt to behavioral actor's theta
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)  # Clip dem gradients
        agent.actor_optimizer.step()

        if self.logger is not None:
            if (self.iter + 1) % LOG_EVERY == 0:
                # Actor gradients
                _grads = self.get_param_grads(self.saddpg_agent.actor.parameters())
                _norm = self.compute_param_grad_norm(self.saddpg_agent.actor.parameters(), norm_type=2)
                self.log_scalar(_norm, 'actor_grad_norm', self.iter)

                # Log losses
                self.log_scalar(J.cpu().detach().numpy(), 'actor_loss', self.iter)
                self.log_scalar(critic_loss.cpu().detach().item(), 'critic_loss', self.iter)


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        soft_update(self.saddpg_agent.target_actor, self.saddpg_agent.actor, self.tau)
        soft_update(self.saddpg_agent.target_critic, self.saddpg_agent.critic, self.tau)


    def log_scalar(self, val, tag='scalar', step = None):
        self.logger.add_scalar(tag, val, step)

    def log_histogram(self, vals, tag='scalar', step = None):
        self.logger.add_histogram(tag, vals, step)


    def compute_param_grad_norm(self, params, norm_type = 2):
        total_norm = 0
        parameters = list(filter(lambda p: p.grad is not None, params))
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm


    def get_param_grads(self, params):
        parameters = list(filter(lambda p: p.grad is not None, params))
        pout = []
        for p in parameters:
            pout += np.copy(p.grad.data.cpu().numpy()).reshape(-1).tolist()
        return pout
            
            




