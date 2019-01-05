# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import numpy as np
from agents.ddpg import DDPGAgent
import torch
from utils.utilities import soft_update, batch_to_tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

LOG_EVERY = 100

class MADDPG:
    def __init__(self, n_agents, partial_obs_size, action_size, full_obs_size,
                 discount_factor=0.995, tau=0.01, logger = None):
        super(MADDPG, self).__init__()

        self.n_agents = n_agents
        self.action_size = action_size
        self.partial_obs_size = partial_obs_size
        self.full_obs_size = full_obs_size
        self.logger = logger

        self.maddpg_agent = [DDPGAgent(in_actor = partial_obs_size,out_actor = action_size,
                                       in_critic= full_obs_size + n_agents*action_size,
                                       lr_actor=1.0e-2, lr_critic=1.0e-2),
                             DDPGAgent(in_actor=partial_obs_size, out_actor=action_size,
                                       in_critic=full_obs_size + n_agents * action_size,
                                       lr_actor=1.0e-2, lr_critic=1.0e-2)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def reset(self):
        [agent.reset() for agent in self.maddpg_agent]

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=None):
        """get actions from all agents in the MADDPG object"""
        # actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        actions = []
        for agent, obs in zip(self.maddpg_agent, obs_all_agents):
            a = agent.act(obs, noise)
            actions.append(a)
        return actions

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        for agent, obs in zip(self.maddpg_agent, obs_all_agents):
            target_actions.append(agent.target_act(obs))
        return target_actions

    def learn(self, samples, agent_number):
        """learn """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]

        # Convention: this agent's actions will be added at the end of the action inputs of the Q-function
        batch_size = len(samples)
        this_agent_partial_obs = []
        all_agents_partial_obs = []
        full_obs = []
        this_agent_action = []
        all_actors_actions = []  # Needed for local Q
        reward = []
        next_partial_obs = []
        next_partial_obs_all_actors = []  # Needed for local Q'
        next_full_obs = []
        done = []
        # other_actors_indeces = np.logical_not(np.arange(0, self.n_agents) == agent_number)

        for e in samples:
            this_agent_partial_obs.append(e[0][agent_number, ...])
            all_agents_partial_obs.append(e[0])  # sample x actor x obs size
            full_obs.append(e[1])
            this_agent_action.append(e[2][agent_number, ...])
            all_actors_actions.append((e[2]))  # sample x actor x act size
            reward.append(e[3][agent_number, ...][None])
            next_partial_obs.append(e[4][agent_number, ...])
            next_partial_obs_all_actors.append(e[4])  # sample x actor x obs size
            next_full_obs.append(e[5])
            done.append(e[6][agent_number, ...][None].astype(np.float32))

        if (self.logger is not None) and ((self.iter + 1) % LOG_EVERY == 0):
            self.logger.add_histogram('rewards_batch', np.array(reward), global_step=self.iter)
            self.logger.add_histogram('full_obs_batch', np.array(full_obs), global_step=self.iter)
            self.logger.add_histogram('all_actors_actions_batch', np.array(all_actors_actions), global_step=self.iter)

        this_agent_partial_obs = torch.tensor(this_agent_partial_obs, dtype=torch.float32).to(device)
        all_agents_partial_obs = torch.tensor(all_agents_partial_obs, dtype=torch.float32).to(device)
        full_obs = torch.tensor(full_obs, dtype=torch.float32).to(device)
        this_agent_action = torch.tensor(this_agent_action, dtype=torch.float32).to(device)
        all_actors_actions = torch.tensor(all_actors_actions, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_partial_obs = torch.tensor(next_partial_obs, dtype=torch.float32).to(device)
        next_partial_obs_all_actors = torch.tensor(next_partial_obs_all_actors, dtype=torch.float32).to(device)
        next_full_obs = torch.tensor(next_full_obs, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        if self.logger is not None:
            if (self.iter + 1) % LOG_EVERY == 0:
                # Log samples
                self.log_histogram(reward, 'rewards', self.iter)
                self.log_histogram(all_actors_actions, 'actions', self.iter)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # reward = (reward - reward.mean())/(reward.std() + 1e-8)

        # -------------- Critic Update ---------------- #
        # -- Target --
        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network

        # Since the gradient of the loss is with wrt to parameters of the behavior actor,
        # based on the deterministic gradient formulation, Grad-Loss(theta) = Grad(Q)*Grad(pi).
        # So the gradients of the loss are with respect to the parameters of the behavior actor,
        # but they flow through the behavior Q-function and beahavior actor.
        # We should not compute the gradient wrt to the parameters of the targets, either
        # critic or actor
        # Note: we need to generate target actions from all actors ... refer to (6)
        # target_actions = agent.target_act(next_obs)
        # target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)

        # Q(s_next, a_next)
        with torch.no_grad():
            # -- GT: The target parameters computed gradients in the original formulation.
            #        But in the paper (6), the loss is wrt to the behavioral parameters, not target
            #        parameters, so computing the gradient wrt them (targets) should not happen
            #        as the application of the gradient descend would update the target parameters
            #        which we should be doing via the soft-update only.

            # Iterate over all target actors to generate target actions for all of them
            # Q*(x', a1', a2', ..., aN')
            # Convention: Q(x, a1, ..). So add state first
            next_target_critic_input = []
            next_target_critic_input.append(next_full_obs.view(batch_size, -1))
            for ii in range(self.n_agents):
                _next_partial_obs = next_partial_obs_all_actors[:,ii,:] # ??
                # _next_partial_obs = all_agents_partial_obs[:, ii, :]
                _next_target_actions = self.maddpg_agent[ii].target_act(_next_partial_obs)
                next_target_critic_input.append(_next_target_actions.view(batch_size, -1))

            next_target_critic_input = torch.cat(next_target_critic_input, dim=1).to(device)
            # --
            q_next = agent.target_critic(next_target_critic_input)

        # y = reward of this timestep + discount * Q'(st_full+1, mu'(st_partial+1))
        y = reward.view(-1, 1) + self.discount_factor * q_next * (1 - done.view(-1, 1))

        # Expected Q: Q(states, actions)
        # Collect the current full state + actions from all agents into list form
        # Convention: Q(x, a1, ..). So add state first
        critic_input = []
        critic_input.append(full_obs)
        for ii in range(self.n_agents):
            critic_input.append(all_actors_actions[:, ii, :])
        critic_input = torch.cat(critic_input, dim=1).to(device)

        # This is the expected Q under theta (the policy which generate the samples, the off policy). The gradient
        # of this loss will be with respect to the parameters of the Q-parameters as they are under behavioral policy now
        q_expected = agent.critic(critic_input)

        # TD error
        huber_loss = torch.nn.SmoothL1Loss()
        # l2_loss = torch.nn.MSELoss()
        critic_loss = huber_loss(q_expected, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)  # necessary ?
        agent.critic_optimizer.step()

        if self.logger is not None:
            if (self.iter + 1) % LOG_EVERY == 0:
                # Critic gradients
                _grads = self.get_param_grads(agent.critic.parameters())
                _norm = self.compute_param_grad_norm(agent.critic.parameters(), norm_type=2)
                self.log_scalar(_norm, 'critic_grad_norm', self.iter)


        # -------------- Actor Update ---------------- #
        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative.
        # we only need the gradient wrt to this agent's policy parameters !!
        # Convention: Q(x, a1, ..). So add state first
        q_input = []
        q_input.append(full_obs)
        for ii in range(self.n_agents):
            # Careful: Per (5), Q(x, a1, a2, ..., aN), ai = mu(oi), while aj (i!=j) are the sampled actions ??!!
            if ii != agent_number:
                _act = all_actors_actions[:, ii, :]  # Use the sample actions for the other actors
                q_input.append(_act.detach())  # Do not track the gradient from the other actors
            else:
                _act = self.maddpg_agent[ii].actor(all_agents_partial_obs[:, ii, :])
                q_input.append(_act)  # Compute the gradient wrt "agent_number" agent
        q_input = torch.cat(q_input, dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input).mean()
        actor_loss.backward()  # wrt to "agent_number" agent
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

        if self.logger is not None:
            if (self.iter + 1) % LOG_EVERY == 0:
                # Actor gradients
                _grads = self.get_param_grads(agent.actor.parameters())
                _norm = self.compute_param_grad_norm(agent.actor.parameters(), norm_type=2)
                self.log_scalar(_norm, 'actor_grad_norm', self.iter)

                # Log losses
                self.log_scalar(actor_loss.cpu().detach().numpy(), 'actor_loss', self.iter)
                self.log_scalar(critic_loss.cpu().detach().item(), 'critic_loss', self.iter)



    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)


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
            
            
            




