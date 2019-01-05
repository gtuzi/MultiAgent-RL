import torch.nn as nn
class NormLayerNet(nn.Module):
    def __init__(self, input_size):
        super(NormLayerNet, self).__init__()
        self.normlayer = nn.BatchNorm1d(input_size)
        self.normlayer.weight.data.fill_(1)
        self.normlayer.bias.data.fill_(0)

    def forward(self, x):
        _size = x.size()
        if len(_size) == 1:
            x = x.unsqueeze(0)
        x = self.normlayer(x)
        return x.view(_size)