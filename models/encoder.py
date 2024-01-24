import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch.nn import BatchNorm1d, Identity
import torch.nn as nn
from utils.register import register


def get_activation(name: str):
        activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
        return activations[name]
    

@register.encoder_register
class GCN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(GCN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden_size)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden_size, hidden_size))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCNConv(hidden_size, hidden_size))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden_size))
                else:
                    self.bns.append(Identity())

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, hidden_size)) 
            # glorot(self.convs[-1].weight)
            if use_bn:
                self.bns.append(BatchNorm1d(hidden_size))
            else:
                self.bns.append(Identity())
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.bns[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            if self.use_bn:
                self.bns[i].reset_parameters()
                

@register.encoder_register
class GIN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(GIN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        
        self.readout = global_mean_pool
        if self.layer_num > 1:
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size)))) 
            for i in range(layer_num-1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size))))
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden_size))
                else:
                    self.bns.append(Identity())

        else: 
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_size),
                                               nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                               nn.Linear(hidden_size, hidden_size)))) 
            if use_bn:
                self.bns.append(BatchNorm1d(hidden_size))
            else:
                self.bns.append(Identity())
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.bns[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
            else:
                x = self.activation(x)
            x = self.dropout(x)

        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            if self.use_bn:
                self.bns[i].reset_parameters()
                

@register.encoder_register
class GAT_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(GAT_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GATConv(input_dim, hidden_size)) 
            for i in range(layer_num-1):
                self.convs.append(GATConv(hidden_size, hidden_size))
            self.bns.append(BatchNorm1d(hidden_size))
        else: 
            self.convs.append(GATConv(input_dim, hidden_size)) 
            self.bns.append(BatchNorm1d(hidden_size))
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.bns[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()
            

@register.encoder_register
class MLP_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(MLP_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        
        self.readout = global_mean_pool
        if self.layer_num > 1:
            self.convs.append(nn.Linear(input_dim, hidden_size)) 
            for i in range(layer_num-1):
                self.convs.append(nn.Linear(hidden_size, hidden_size))
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden_size))
                else:
                    self.bns.append(Identity())

        else: 
            self.convs.append(nn.Linear(input_dim, hidden_size))
            if use_bn:
                self.bns.append(BatchNorm1d(hidden_size))
            else:
                self.bns.append(Identity())
    
    def forward(self, x, edge_index, **kwargs):
        for i in range(self.layer_num):
            x = self.bns[i](self.convs[i](x))
            if i == self.layer_num - 1 and not self.last_act:
                pass
            else:
                x = self.activation(x)
            x = self.dropout(x)
            
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            if self.use_bn:
                self.bns[i].reset_parameters()