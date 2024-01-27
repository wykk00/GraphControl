from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from utils.register import register
import copy
from .gcc import GCC

@register.model_register
class GCC_GraphControl(nn.Module):

    def __init__(
        self,
        **kwargs
    ):
        super(GCC_GraphControl, self).__init__()
        input_dim = kwargs['positional_embedding_size']
        hidden_size = kwargs['node_hidden_dim']
        output_dim = kwargs['num_classes']
        
        self.encoder = GCC(**kwargs)
        self.trainable_copy = copy.deepcopy(self.encoder)
        
        self.zero_conv1 = torch.nn.Linear(input_dim, input_dim)     
        self.zero_conv2 = torch.nn.Linear(hidden_size, hidden_size)

        self.linear_classifier = torch.nn.Linear(hidden_size, output_dim)

        with torch.no_grad():
            self.zero_conv1.weight = torch.nn.Parameter(torch.zeros(input_dim, input_dim))
            self.zero_conv1.bias = torch.nn.Parameter(torch.zeros(input_dim))
            self.zero_conv2.weight = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.zero_conv2.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        
        self.prompt = torch.nn.Parameter(torch.normal(mean=0, std=0.01, size=(1, input_dim)))
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        raise NotImplementedError('Please use --subsampling')
    
    def reset_classifier(self):
        self.linear_classifier.reset_parameters()
    
    def forward_subgraph(self, x, x_sim, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

            x_down = self.zero_conv1(x_sim)
            x_down = x_down + x
            
            # for simplicity, we use edge_index to calculate degrees
            x_down = self.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)
            
            x_down = self.zero_conv2(x_down)
            
            out = x_down + out
        else:
            raise NotImplementedError('Please freeze pre-trained models')
        
        x = self.linear_classifier(out)
        return x