from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, name, dataset, device, num_nodes, seq_len, horizon, input_dim, output_dim):
        super(BaseModel, self).__init__()
        self.name = name
        self.dataset = dataset
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError
    
    def param_num(self, str):
        return sum([param.nelement() for param in self.parameters()])

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)