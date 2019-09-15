import torch
from torch import nn
import torch.nn.functional as F
import utils

class Conv2D(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size, mask_type = None):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        if mask_type is not None:
            self.mask = mask(filter_size, input_dim, output_dim, mask_type)
        
        self.weights = nn.Parameter(nn.init(torch.empty(output_dim, input_dim, filter_size, filter_size)))

    def forward(self, inputs):
        
        weights = matmul(self.weights, self.mask)
        output = F.conv2d(inputs, weights)

        return output


class Conv1D(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size, mask_type = None):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        if mask_type is not None:
            self.mask = mask(filter_size, input_dim, output_dim, mask_type)
        
        self.weights = nn.Parameter(nn.init(torch.empty(output_dim, input_dim, filter_size)))

    def forward(self, inputs):
        
        weights = matmul(self.weights, self.mask)
        output = F.conv1d(inputs, weights)

        return output
