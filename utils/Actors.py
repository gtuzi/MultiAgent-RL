import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, obs_size, action_size, fc_units=(256, 256), norm_input_layer = None):
        """Initialize parameters and build model.
        Params
        ======
            obs_size (int): Dimension of each observation
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        c_sizes = (obs_size,) + fc_units + (action_size,)
        self.n_layers = len(c_sizes) - 1

        self.fc1 = nn.Linear(obs_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)
        self.reset_parameters()

        self.norm_input_layer = norm_input_layer

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        """Build an actor (policy) network that maps states -> actions."""
        obs = obs.contiguous()
        if self.norm_input_layer is not None:
            obs = self.norm_input_layer(obs)

        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class ActorNormLayers(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc_units=(256, 256)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNormLayers, self).__init__()
        c_sizes = (state_size, ) + fc_units + (action_size,)
        self.n_layers = len(c_sizes) - 1

        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)

        self.fcnorm1 = nn.LayerNorm(fc_units[0])
        self.fcnorm2 = nn.LayerNorm(fc_units[1])

        self.reset_parameters()
        self.norm_input_layer = nn.BatchNorm1d(state_size) # norm_input_layer
        self.norm_input_layer.weight.data.fill_(1)
        self.norm_input_layer.bias.data.fill_(0)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        """Build an actor (policy) network that maps states -> actions."""
        obs = obs.contiguous()
        if self.norm_input_layer is not None:
            if len(list(obs.size())) == 1:
                obs = self.norm_input_layer(obs.unsqueeze(0)).squeeze(0)
            else:
                obs = self.norm_input_layer(obs)

        x = F.leaky_relu(self.fcnorm1(self.fc1(obs)))
        x = F.leaky_relu(self.fcnorm2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x


class DiscreteActorNormLayers(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc_units=(256, 256)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DiscreteActorNormLayers, self).__init__()
        c_sizes = (state_size, ) + fc_units + (action_size,)
        self.n_layers = len(c_sizes) - 1

        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)

        self.fcnorm1 = nn.LayerNorm(fc_units[0])
        self.fcnorm2 = nn.LayerNorm(fc_units[1])

        self.reset_parameters()
        self.norm_input_layer = nn.BatchNorm1d(state_size) # norm_input_layer
        self.norm_input_layer.weight.data.fill_(1)
        self.norm_input_layer.bias.data.fill_(0)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        """Build an actor (policy) network that maps states -> actions."""
        obs = obs.contiguous()
        if self.norm_input_layer is not None:
            if len(list(obs.size())) == 1:
                obs = self.norm_input_layer(obs.unsqueeze(0)).squeeze(0)
            else:
                obs = self.norm_input_layer(obs)

        x = F.leaky_relu(self.fcnorm1(self.fc1(obs)))
        x = F.leaky_relu(self.fcnorm2(self.fc2(x)))
        logit = self.fc3(x)
        log_p = F.log_softmax(logit)
        return logit, log_p

