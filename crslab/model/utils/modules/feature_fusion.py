import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F


class Fusion2_GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(Fusion2_GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)

        return gated_emb

class Fusion2_MinusFCLayer(nn.Module):
    def __init__(self, input_dim):
        super(Fusion2_MinusFCLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 3, input_dim)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2, input1-input2], dim=-1)) # (bs, dim)

        return norm_input

class Fusion3_FCLayer(nn.Module):
    def __init__(self, input_dim):
        super(Fusion3_FCLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 3, input_dim)

    def forward(self, input1, input2, input3):
        norm_input = self._norm_layer1(torch.cat([input1, input2, input3], dim=-1)) # (bs, dim)

        return norm_input

class Fusion2_FCLayer(nn.Module):
    def __init__(self, input_dim):
        super(Fusion2_FCLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1)) # (bs, dim)

        return norm_input

class Fusion3_MinusFCLayer(nn.Module):
    def __init__(self, input_dim):
        super(Fusion3_MinusFCLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 6, input_dim)

    def forward(self, input1, input2, input3):
        norm_input = self._norm_layer1(torch.cat(
            [input1, input2, input3, input1 - input2, input1 - input3, input2 - input3], 
            dim=-1
        )) # (bs, dim)

        return norm_input