import torch
import torch.nn
import torch.nn.functional as F
from conv_layers import Conv2D, Conv1D


#implement a function for various initializations of the cell and hidden states

class diagonal_LSTM(nn.Module):

    def __init__(self, input_dim, is_filter_size, ss_filter_size):
        
        super(self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim

        self.input_to_state = Conv2D(input_dim, input_dim*4, is_filter_size, "B")
        self.state_to_state = Conv1D(input_dim, input_dim*4, ss_filter_size)

    def forward(self, inputs):
        
        inputs = skew(inputs)
        i_s = input_to_state(inputs) 
        
        def cell(present_i_s, hidden):
            h_prev, c_prev = hidden

            h_prev = torch.cat([torch.zeros((get_size(h_prev)[:-1]).append(1)), h_prev], dim = 2, out = h_prev)
            s_s = self.state_to_state(h_prev)

            o_f_i, g = torch.split(present_i_s + s_s, [3*hidden_dim, hidden_dim], 1)
            o, f, i = torch.split(torch.nn.sigmoid(o_f_i), hidden_dim, 1)
            g = F.tanh(g)
            
            c = (f * c_prev) + (i * g)
            h = o * F.tanh(c)

            return (h, c)

        output = []
        steps = range(get_size(inputs)[3])
        hidden = self.init_hidden(get_size(inputs)[:-1])
        for i in steps:
            hidden = cell(inputs[:, :, :, i], hidden)
            output.append(hidden[0])

        output = torch.stack(output, dim = 3, out = 'output')

        return unskew(output)

    def init_hidden(self, shape):
        h0 = Variable(torch.zeros(shape))
        c0 = Variable(torch.zeros(shape))

        return (h0, c0)
