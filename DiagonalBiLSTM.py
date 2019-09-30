import torch
import torch.nn as nn
from utils import *
from DiagonalLSTM import diagonal_LSTM

class Diagonal_BiLSTM(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.forward_LSTM = diagonal_LSTM(input_dim, (1, 1), (2, 1))
        self.backward_LSTM = diagonal_LSTM(input_dim, (1,1), (2, 1))

    def __call__(self, inputs):

        forward = self.forward_LSTM(inputs)
        backward = self.backward_LSTM(torch.flip(inputs, dims = [3]))

        backward = torch.flip(backward, dims = [3])

        size = get_size(backward)
        s = torch.zeros(size[0], size[1], 1, size[3]).to(backward.get_device())
        backward = torch.cat((s, backward[:, :, :-1, :]), dim = 2)
        output = forward + backward
        
        return output

    
