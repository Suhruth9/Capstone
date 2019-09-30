import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Conv2D(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size, mask_type = None, padding = (0, 0)):
        super().__init__()        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        self.padding = padding
        if mask_type is not None:
            self.mask = mask(filter_size, input_dim, output_dim, mask_type)
        
        self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty(output_dim, input_dim, filter_size[0], filter_size[1])))
        
        
    def forward(self, inputs):
        device = inputs.get_device()
        mask = self.mask.to(device)
        weights = self.weights.to(device)
        weights = torch.mul(weights, mask)
        
        output = F.conv2d(inputs, weights, padding = self.padding)

        return output


class Conv1D(nn.Module):

    def __init__(self, input_dim, output_dim, filter_size, mask_type = None):
        super().__init__()        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        self.mask_type = mask_type
        if mask_type is not None:
            mask = mask(filter_size, input_dim, output_dim, mask_type)
            
        
        self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty(output_dim, input_dim, filter_size[0], filter_size[1])))

    def forward(self, inputs):
        inputs = torch.unsqueeze(inputs, dim = 3)
##        inputs = inputs.to("cuda:2")
        if self.mask_type != None:
            device = inputs.get_device()
            mask = self.mask.to(device)
            weights = self.weights.to(device)
            weights = torch.mul(weights, self.mask)
            
            output = F.conv2d(inputs, weights)
  
        else:
            device = inputs.get_device()
            weights = self.weights.to(device)
            output = F.conv2d(inputs, weights)

        output = torch.squeeze(output)
        return output
