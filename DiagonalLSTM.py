import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_layers import Conv2D, Conv1D
from utils import *
from torch.autograd import Variable


#implement a function for various initializations of the cell and hidden states

class diagonal_LSTM(nn.Module):

    def __init__(self, input_dim, is_filter_size, ss_filter_size):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim

        self.input_to_state = Conv2D(input_dim, input_dim*4, is_filter_size, "B")
        self.state_to_state = Conv1D(input_dim, input_dim*4, ss_filter_size)

    def forward(self, inputs):
        device = inputs.get_device()
        inputs = skew_features(inputs)
        i_s = self.input_to_state(inputs) 
        
        def cell(present_i_s, hidden, device):
            h_prev, c_prev = hidden
            hidden_dim = self.hidden_dim
            s = (get_size(h_prev))[:-1]
            s = torch.zeros(s)
            s = torch.unsqueeze(s, dim = 2).to(device)

            
            
            h_prev = torch.cat((s, h_prev), dim = 2)
            s_s = self.state_to_state(h_prev)

            o_f_i, g = torch.split(present_i_s + s_s, [3*hidden_dim, hidden_dim], 1)
            o, f, i = torch.split(torch.sigmoid(o_f_i), hidden_dim, 1)
            g = torch.tanh(g)

            c = (f * c_prev) + (i * g)
            h = o * torch.tanh(c)
            
            return (h, c)

        output = []
        steps = range(get_size(i_s)[3])
        
        hidden = self.init_hidden(get_size(inputs)[:-1], device)

        for i in steps:
            hidden = cell(i_s[:, :, :, i], hidden, device)
            output.append(hidden[0])

        output = torch.stack(output, dim = 3)
        
        return unskew_features(output)

    def init_hidden(self, shape, device):
        h0 = Variable(torch.zeros(shape)).to(device)
        c0 = Variable(torch.zeros(shape)).to(device)

        return (h0, c0)
