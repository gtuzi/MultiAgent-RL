import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, fc_hidden_units=(256, 256)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, fc_hidden_units[0])
        self.fc2 = nn.Linear(fc_hidden_units[0], fc_hidden_units[1])
        self.fc3 = nn.Linear(fc_hidden_units[1], 1)
        self.reset_parameters()
        self.norm_input_layer = nn.BatchNorm1d(input_size) # norm_input_layer
        self.norm_input_layer.weight.data.fill_(1)
        self.norm_input_layer.bias.data.fill_(0)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.fill_(1.)

    def forward(self, x):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = x.contiguous()
        if self.norm_input_layer is not None:
            if len(list(x.size())) == 1:
                x = self.norm_input_layer(x.unsqueeze(0)).squeeze(0)
            else:
                x = self.norm_input_layer(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class DiscreteCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, out_size, fc_hidden_units=(256, 256)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(DiscreteCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, fc_hidden_units[0])
        self.fc2 = nn.Linear(fc_hidden_units[0], fc_hidden_units[1])
        self.fc3 = nn.Linear(fc_hidden_units[1], out_size)
        self.reset_parameters()
        self.norm_input_layer = nn.BatchNorm1d(input_size) # norm_input_layer
        self.norm_input_layer.weight.data.fill_(1)
        self.norm_input_layer.bias.data.fill_(0)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.fill_(1.)

    def forward(self, x):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = x.contiguous()
        if self.norm_input_layer is not None:
            if len(list(x.size())) == 1:
                x = self.norm_input_layer(x.unsqueeze(0)).squeeze(0)
            else:
                x = self.norm_input_layer(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)