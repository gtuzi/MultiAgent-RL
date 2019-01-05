# individual network settings for each actor + critic pair
# see networkforall for details

from utils.Actors import ActorNormLayers
from utils.Critics import Critic
from utils.NormLayer import NormLayerNet
from utils.utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGPSNEAgent:
    def __init__(self, in_actor,out_actor,in_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGPSNEAgent, self).__init__()
        self.action_size = in_actor

        self.actor = ActorNormLayers(in_actor, out_actor).to(device)
        self.perturbed_actor = ActorNormLayers(in_actor, out_actor).to(device)
        self.target_actor = ActorNormLayers(in_actor, out_actor).to(device)

        self.critic = Critic(in_critic).to(device)
        self.target_critic = Critic(in_critic).to(device)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(list(self.actor.parameters()), lr=lr_actor)
        self.critic_optimizer = Adam(list(self.critic.parameters()), lr=lr_critic, weight_decay=1.e-5)

        self.perturb_parm_sigma = 0.2
        self.perturb_delta = 0.2
        self.perturb_alpha = 1.01
        # Initialize the perturbed actor
        self.__perturb_the_actor__()


    def act(self, obs, use_perturbed_actor=True, noise_delta = 0.2):
        obs = obs.to(device)
        with torch.no_grad():
            if use_perturbed_actor:
                self.perturb_delta = noise_delta
                mode = self.perturbed_actor.training
                self.perturbed_actor.eval() # We want an act here. So batch norm-layers need to be in eval
                action = self.perturbed_actor(obs)
                self.perturbed_actor.train(mode)
            else:
                mode = self.actor.training
                self.actor.eval()
                action = self.actor(obs)
                self.actor.train(mode)
        return torch.clamp(action, -1., 1.)


    def target_act(self, obs):
        obs = obs.to(device)
        with torch.no_grad():
            mode = self.target_actor.training
            self.target_actor.eval()
            action = self.target_actor(obs)
            self.target_actor.train(mode)
        return action


    ################################ Parameter Perturbation ###############################

    def __perturb_the_actor__(self):
        # Copy the local actor, it's time to boogie
        self.perturbed_actor.load_state_dict(self.actor.state_dict())
        for n, p in dict(self.perturbed_actor.named_parameters()).items():
            # if n.split('.')[0] not in ['fc1', 'fc2', 'fc3']:
            if n.split('.')[0] not in ['norm_input_layer']:
                datasize = p.data.size()
                nels = p.data.numel()
                e = torch.distributions.normal.Normal(torch.zeros(nels), self.perturb_parm_sigma * torch.ones(nels)).sample()
                p.data.add_(e.view(datasize).to(device))

    def perturb_the_actor(self, observations):
        '''
            Perturbing the parameters per paper
        :return:
        '''

        self.__perturb_the_actor__()
        # Compute the distance that the perturbation introduces
        d = self._compute_sample_perturbed_distance(observations)
        self._adapt_perturbation_sigma(d)
        self.last_d = d
        self.las_obs = observations

    def _compute_sample_perturbed_distance(self, states):
        mode = self.actor.training
        self.actor.eval()
        self.perturbed_actor.eval()

        a = self.actor(states)
        ap = self.perturbed_actor(states)

        self.actor.train(mode)
        self.perturbed_actor.train(mode)

        d = (1. / self.action_size) * torch.mean((a - ap) ** 2)
        return torch.sqrt(d)

    def _adapt_perturbation_sigma(self, d):
        if d < self.perturb_delta:
            self.perturb_parm_sigma *= self.perturb_alpha
        else:
            self.perturb_parm_sigma *= 1. / self.perturb_alpha

        ##########################################################################################
