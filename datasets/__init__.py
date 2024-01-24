from torch_geometric.utils import to_undirected, homophily
import torch_geometric.transforms as T
import copy
import torch
import os
import numpy as np

from .dataset import Amazon, Coauthor, Airports, CitationFull, HindexDataset
from utils.random import reset_random_seed
from utils.transforms import obtain_attributes


dataset_dict = {
    'Photo': Amazon,
    'Physics': Coauthor,
    'usa': Airports,
    'brazil': Airports,
    'europe': Airports,
    'DBLP': CitationFull,
    'Cora_ML': CitationFull,
    'Hindex': HindexDataset
}

PATH = './datasets/data'

def load_dataset(dataset_name, trans=None):
    if dataset_name in ['Hindex']:
        if trans == None:
            return dataset_dict[dataset_name](root=f'{PATH}/{dataset_name}')
        else:
            return dataset_dict[dataset_name](root=f'{PATH}/{dataset_name}', transform=T.Compose([trans]))
    else:
        if trans == None:
            return dataset_dict[dataset_name](root=PATH, name=dataset_name)
        else:
            return dataset_dict[dataset_name](root=PATH, name=dataset_name, transform=T.Compose([trans]))
    

class NodeDataset:
    def __init__(self, dataset_name, trans=None, n_seeds=[0]) -> None:
        self.path = PATH
        self.dataset_name = dataset_name
        if dataset_name in ['Hindex']:
            self.dataset = dataset_dict[dataset_name](root=f'{self.path}/{dataset_name}', transform=trans)
        else:
            self.dataset = dataset_dict[dataset_name](root=f'{self.path}', name=dataset_name, transform=trans)

        self.num_classes = self.dataset.num_classes
        self.num_node_features = self.dataset.num_node_features  
        
        assert len(self.dataset) == 1, "Training data consists of multiple graphs!"

        self.data = self.dataset[0]

        # parse it into undirected graph
        edge_index = to_undirected(self.data.edge_index)
        self.data.edge_index = edge_index
        self.num_nodes = self.data.x.shape[0]
        
        # backup original node attributes and edges
        self.backup_x = copy.deepcopy(self.data.x)
        self.backup_edges = copy.deepcopy(self.data.edge_index)
        self.random_split = False
        
        # For datasets without node attributes, we will use node embeddings from Node2Vec as their node attributes
        attr_path = f'{PATH}/{dataset_name}/processed/node2vec.pt'
        if dataset_name in ['USA', 'Europe', 'Brazil', 'Hindex'] and os.path.exists(attr_path):
            x = torch.load(attr_path)
            self.data.x = x.detach()

        # If the dataset does not contain preset splits, we will randomly split it into train:test=1:9 twenty times
        if not hasattr(self.data, 'train_mask'):
            self.random_split = True
            num_train = int(self.num_nodes*0.1)
            
            train_mask_list = []
            test_mask_list = []
            for seed in n_seeds:
                reset_random_seed(seed)

                rand_node_idx = torch.randperm(self.num_nodes)
                train_idx = rand_node_idx[:num_train]
                train_mask = torch.zeros(self.num_nodes).bool()
                train_mask[train_idx] = True
                
                test_mask = torch.ones_like(train_mask).bool()
                test_mask[train_idx] = False
                train_mask_list.append(train_mask.unsqueeze(1))
                test_mask_list.append(test_mask.unsqueeze(1))
            
            self.data.train_mask = torch.cat(train_mask_list, dim=1)
            self.data.test_mask = torch.cat(test_mask_list, dim=1)


    def generate_subgraph(self):
        pass
    
    def split_train_test(self, split_ratio=0.8):
        raise NotImplementedError('do not set parameter <split>')
    
    def to(self, device):
        self.data = self.data.to(device)
    
    def replace_node_attributes(self, use_adj, threshold, num_dim):
        self.num_node_features = num_dim
        self.data.x = obtain_attributes(self.data, use_adj, threshold, num_dim)
        
    def obtain_node_attributes(self, use_adj, threshold=0.1, num_dim=32):
        return obtain_attributes(self.data, use_adj, threshold, num_dim)
    
    def print_statistics(self):
        h = homophily(self.data.edge_index, self.data.y)
        from collections import Counter
        if len(self.data.y.shape) >= 2: # For one-hot labels
            y = self.data.y.argmax(1)
        else:
            y = self.data.y
        count = Counter(y.tolist())
        total_num = sum(count.values())
        class_ratio = {}
        for key, value in count.items():
            r = round(value / total_num, 2)
            class_ratio[key] = r
        print(f'{self.dataset_name}: Number of nodes: {self.num_nodes}, Dimension of features: {self.num_node_features}, Number of edges: {self.data.edge_index.shape[1]}, Number of classes: {self.num_classes}, Homophily: {h}, Class ratio: {class_ratio}.')
        if self.random_split:
            print('The dataset does not contain preset splits, we randomly split the dataset twenty times. Train: teset = 1:9')
        else:
            print('We use the preset splits.')
        

if __name__ == '__main__':
    dataset = NodeDataset('Hindex')
    print(dataset)
