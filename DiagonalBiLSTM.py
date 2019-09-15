import torch
from torch import nn
from utils import *
from DiagonalLSTM import diagonal_LSTM

class Diagonal_BiLSTM():

    def __init__(self, input_dim):

        self.forward_LSTM = diagonal_LSTM(input_dim, (1, 1), (2, 1))
        self.backward_LSTM = diagonal_LSTM(input_dim, (1,1), (2, 1))

    def __call__(self, inputs):

        forward = forward_LSTM(inputs)
        backward = backward_LSTM(torch.flip(inputs, dims = [3]))

        backward = torch.flip(backward, dims = [3])

        size = get_size(backward)
        backward = torch.cat((torch.zeros(size[0], size[1], 1, size[3]), backwards[:, :, -1, :]), dim = 2)

        output = forward + backward

        return output

    
