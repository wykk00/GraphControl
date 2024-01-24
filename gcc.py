import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import ShaDowKHopSampler, DataLoader

from utils.random import reset_random_seed
from utils.args import Arguments
from models import load_model
from datasets import NodeDataset
from utils.transforms import process_attributes
from utils.sampling import ego_graphs_sampler, collect_subgraphs


def preprocess(config, dataset_obj):
    kwargs = {'batch_size': config.batch_size, 'num_workers': 3, 'persistent_workers': True}

    print('generating subgraphs....')

    train_loader, test_loader = None, None
    
    train_idx = dataset_obj.data.train_mask.nonzero().squeeze()
    test_idx = dataset_obj.data.test_mask.nonzero().squeeze()
    
    train_graphs = collect_subgraphs(train_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)
    test_graphs = collect_subgraphs(test_idx, dataset_obj.data, walk_steps=config.walk_steps, restart_ratio=config.restart)

    if config.use_adj:
        [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in train_graphs]
        [process_attributes(g, use_adj=config.use_adj, threshold=config.threshold, num_dim=config.num_dim) for g in test_graphs]
    
    dataset_obj.num_node_features = config.num_dim
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)
    
    return train_loader, test_loader


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    dataset_obj = NodeDataset(config.dataset, n_seeds=config.seeds)
    dataset_obj.print_statistics()
    
    acc_list = []
    
    train_masks = dataset_obj.data.train_mask
    test_masks = dataset_obj.data.test_mask

    for _, seed in enumerate(config.seeds):
        reset_random_seed(seed)

        if dataset_obj.random_split:
            dataset_obj.data.train_mask = train_masks[:, seed]
            dataset_obj.data.test_mask = test_masks[:, seed]
        
        train_loader, test_loader = preprocess(config, dataset_obj)
        model = load_model(dataset_obj.num_node_features, dataset_obj.num_classes, config).to(device)

        # training model
        train_subgraph(config, model, train_loader, device)
        acc = eval_subgraph(config, model, test_loader, device)
        
        acc_list.append(acc)
        print(f'Seed: {seed}, Accuracy: {acc:.4f}')

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
        
def train_subgraph(config, model, train_loader, device):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in tqdm(range(config.epochs)):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if not hasattr(batch, 'root_n_id'):
                batch.root_n_id = batch.root_n_index
            # sign flip, because the sign of eigen-vectors can be filpped randomly (annotate this operate if we conduct eigen-decomposition on full graph)
            sign_flip = torch.rand(batch.x.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch.x = batch.x*sign_flip.unsqueeze(0)
            
            out = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_id)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()


def eval_subgraph(config, model, test_loader, device):
    model.eval()
    
    correct = 0
    total_num = 0
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, 'root_n_id'):
            batch.root_n_id = batch.root_n_index
        
        preds = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_id).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc

if __name__ == '__main__':
    config = Arguments().parse_args()
    
    main(config)