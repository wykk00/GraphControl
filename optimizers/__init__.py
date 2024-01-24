import torch


optimizers_dicts = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'radam': torch.optim.RAdam,
    'nadam': torch.optim.NAdam
}

def create_optimizer(**kwargs):
    lr = kwargs['lr']
    weight_decay =  kwargs['weight_decay']
    name = kwargs['name']
    parameters = kwargs['parameters']
    
    return optimizers_dicts[name](parameters, lr=lr, weight_decay=weight_decay)
    