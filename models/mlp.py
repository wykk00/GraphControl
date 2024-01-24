import torch
from torch import nn
from utils.register import register
from .encoder import MLP_Encoder
from torch_geometric.nn import global_mean_pool


class Two_MLP_BN(torch.nn.Module):
    r"""
    Applies a non-linear transformation to contrastive space from representations.

        Args:
            hidden size of encoder, mlp hidden size, mlp output size
    """
    def __init__(self, hidden, mlp_hid, mlp_out):

        super(Two_MLP_BN, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, mlp_hid), 
            nn.BatchNorm1d(mlp_hid),
            nn.ReLU(),
            nn.Linear(mlp_hid, mlp_out)
        )

    def forward(self, feat):
        return self.proj(feat)
    
class Two_MLP(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                
                


@register.model_register
class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=70, activation="relu", dropout=0.5, use_bn=False, **kargs):
        super(MLP, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim

        self.encoder = MLP_Encoder(input_dim, layer_num, hidden_size, activation, dropout, use_bn)
        
        self.eigen_val_emb = torch.nn.Sequential(torch.nn.Linear(32, hidden_size), 
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_size, hidden_size))

        self.classifier = torch.nn.Linear(hidden_size, output_dim)
        self.linear_classifier = torch.nn.Linear(hidden_size*2, output_dim)
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
            
        x = self.classifier(x)
        return x
    
    def forward_subgraph(self, x, edge_index, batch, root_n_id, edge_weight=None, **kwargs):
        x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        
        x = self.linear_classifier(x) # use linear classifier
        return x
    
    def reset_classifier(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        

