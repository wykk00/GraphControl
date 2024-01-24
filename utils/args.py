import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seeds", type=int, nargs="+", default=[0])
        # Dataset
        self.parser.add_argument('--dataset', type=str, help="dataset name", default='Cora_ML')

        # Model configuration
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=128)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.0)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu', 
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        self.parser.add_argument('--use_bn', action='store_true', help="use BN or not")
        self.parser.add_argument('--model', type=str, help="model name", default='GCC_ControlNet', 
                                 choices=['GCC', 'GCC_ControlNet'])
    
        # Training settings
        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam', 
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=5e-4)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=200)
        self.parser.add_argument('--batch_size', type=int, default=128)
        self.parser.add_argument('--finetune', action='store_true', help="Quickly find optim parameters")
        
        # Processing node attributes
        self.parser.add_argument('--use_adj', action='store_true', help="use eigen-vectors of adjacent matrix as node attributes")
        self.parser.add_argument('--threshold', type=float, help="the threshold for discreting similarity matrix", default=0.15)
        self.parser.add_argument('--num_dim', type=int, help="the number of replaced node attributes", default=32)     
        self.parser.add_argument('--ad_aug', action='store_true', help="adversarial augmentation")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.3)
        self.parser.add_argument('--walk_steps', type=int, help="the number of random walk's steps", default=256)

        # Node2vec config
        self.parser.add_argument('--emb_dim', type=int, default=128, help="Embedding dim for node2vec")
        self.parser.add_argument('--walk_length', type=int, default=50, help="Walk length for node2vec")
        self.parser.add_argument('--context_size', type=int, default=10, help="Context size for node2vec")
        self.parser.add_argument('--walk_per_nodes', type=int, default=10, help="Walk per nodes for node2vec")
        
    def parse_args(self):
        return self.parser.parse_args()
