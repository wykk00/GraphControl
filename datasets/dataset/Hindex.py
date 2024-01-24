import torch
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data
from collections import defaultdict
import numpy as np


class HindexDataset(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'Hindex'
        self.root = root
        
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
              

    @property
    def raw_file_names(self):
        return ['aminer_hindex_rand20intop200_5000.edgelist', 'aminer_hindex_rand20intop200_5000.nodelabel']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw') 

    def process(self):
        # Read data into huge `Data` list.
        edge_index, y, self.node2id = self._preprocess(self.raw_paths[0], self.raw_paths[1])
        data = Data(x=torch.zeros(y.size(0), 1), edge_index=edge_index, y=y.argmax(1))
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "Hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "Hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id