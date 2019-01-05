# individual network settings for each actor + critic pair
# see networkforall for details

from utils.Actors import ActorNormLayers
from utils.Critics import Critic
from utils.NormLayer import NormLayerNet
from utils.utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from utils.OUNoise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent:
    def __init__(self, in_actor,out_actor,in_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = ActorNormLayers(in_actor, out_actor).to(device)
        self.target_actor = ActorNormLayers(in_actor, out_actor).to(device)

        self.critic = Critic(in_critic).to(device)
        self.target_critic = Critic(in_critic).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)
        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(list(self.actor.parameters()), lr=lr_actor)
        self.critic_optimizer = Adam(list(self.critic.parameters()), lr=lr_critic, weight_decay=1.e-5)


    def reset(self):
        self.noise.reset()

    def act(self, obs, noise=None):
        obs = obs.to(device)
        with torch.no_grad():
            mode = self.actor.training
            self.actor.eval()
            if noise is not None:
                e = noise * self.noise.noise().to(device)
            else:
                e = 0.0
            action = self.actor(obs) + e
            self.actor.train(mode)

        return torch.clamp(action, -1., 1.)


    def target_act(self, obs):
        obs = obs.to(device)
        with torch.no_grad():
            mode = self.target_actor.training
            self.target_actor.eval()
            action = self.target_actor(obs)
            self.target_actor.train(mode) # As it was before

        return torch.clamp(action, -1., 1.)
