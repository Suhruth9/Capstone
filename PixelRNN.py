import torch
from torch import nn
import torch.nn.functional as F
from base_model import BaseModel
from utils import *
from DialgonalBiLSTM import Diagonal_BiLSTM

class PixelRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_conv2d = Conv2D(1, 16, 7, "A")
        self.init_residual_block()
        self.init_output_conv_block()
        

    def init_residual_block(self):
        for i in range(7):
            setattr(self, 'RNN_layer_{:d}'.format(i+1), DiagonalBILSTM(16))

    def init_output_conv_block(self):
        for i in range(2):
            setattr(self, 'output_conv2d_{:d}'.format(i+1), Conv2D(16, 16, 7, "B"))
        setattr(self, 'output_conv2d', Conv2D(16, 1, 1, "B"))
        
            
    def forward(self, X):
        output = self.input_conv2d(X)

        for i in range(self.opt.rnn_layers):
            RNN_layer = getattr(self, 'RNN_layer_{:d}'.format(i+1))
            output = RNN_layer(output)

        
        for i in range(self.opt.output_conv_layers):
            CNN_layer = getattr(self,'output_conv2d_{:d}'.format(i+1))
            output = CNN_layer(output)
            output = F.relu(output)

        output_layer = getattr(self,'output_conv2d')
        output = output_layer(output)


        return output

        

        
        


        

        
        
        
        
        
        
        
        
    
        
        
        
    
