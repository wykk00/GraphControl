'''
This file is used for generating node embeddings for datasets with graph topology.
'''

import os.path as osp
import sys

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from datasets import NodeDataset
from utils.args import Arguments
from utils.random import reset_random_seed
import torch
import os
import numpy as np
from tqdm import tqdm

PATH = f'./datasets/data'


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
    return acc


if __name__ == "__main__":
    config = Arguments().parse_args()

    dataset_obj = NodeDataset(dataset_name=config.dataset)
    dataset_obj.print_statistics()
    data = dataset_obj.data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0 if sys.platform.startswith('win') else 4
    
    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask
    
    model = Node2Vec(
            data.edge_index,
            embedding_dim=config.emb_dim, # 256 for USA, 64 for Europe, 32 for Brazil
            walk_length=config.walk_length, 
            context_size=config.context_size,
            walks_per_node=config.walk_per_nodes,
            sparse=True,
        ).to(device)
    
    loader = model.loader(batch_size=config.batch_size, shuffle=True,
                        num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config.lr)
    
    if dataset_obj.random_split:
        dataset_obj.data.train_mask = train_masks[:, 0]
        dataset_obj.data.test_mask = test_masks[:, 0]
    
    progress = tqdm(range(0, config.epochs))
    for epoch in progress:
        loss = train()
        acc = test()
        progress.set_postfix_str(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    
    # save embedding
    torch.save(model.embedding.weight.cpu(), f=f'{PATH}/{config.dataset}/processed/node2vec.pt')

    acc_list = []

    for seed in config.seeds:
        reset_random_seed(seed)
        dataset_obj.data.train_mask = train_masks[:, seed]
        dataset_obj.data.test_mask = test_masks[:, seed]
        acc = test()
        acc_list.append(acc)
        
    
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
        