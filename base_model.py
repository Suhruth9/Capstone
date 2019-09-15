import os
from torch import nn
from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC, nn.module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def update():
        pass

    @abstractmethod
    def forward():
        pass

    def load_model():
        pass


    

        
    
