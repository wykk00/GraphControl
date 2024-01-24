import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected, remove_isolated_nodes, dropout_adj, remove_self_loops, k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy
from torch_sparse import SparseTensor

from .transforms import obtain_attributes


def add_remaining_selfloop_for_isolated_nodes(edge_index, num_nodes):
    num_nodes = max(maybe_num_nodes(edge_index), num_nodes)
    # only add self-loop on isolated nodes
    # edge_index, _ = remove_self_loops(edge_index)
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    connected_nodes_indices = torch.cat([edge_index[0], edge_index[1]]).unique()
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[connected_nodes_indices] = False
    loops_for_isolatd_nodes = loop_index[mask]
    loops_for_isolatd_nodes = loops_for_isolatd_nodes.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops_for_isolatd_nodes], dim=1)
    return edge_index


class RWR:
    """ Every node in the graph will get a random path

    A stochastic data augmentation module that transforms a complete graph into many subgraphs through random walking
    the subgraphs which contain the same center nodes are positive pairs, otherwise they are negative pairs
    """

    def __init__(self, walk_steps=50, graph_num=128, restart_ratio=0.5, inductive=False, aligned=False, **args):
        self.walk_steps = walk_steps
        self.graph_num = graph_num
        self.restart_ratio = restart_ratio
        self.inductive = inductive
        self.aligned = aligned

    def __call__(self, graph):
        graph  = copy.deepcopy(graph) # modified on the copy
        assert self.walk_steps > 1
        # remove isolated nodes (or we can construct edges for these nodes)
        if self.inductive:
            train_node_idx = torch.where(graph.train_mask == True)[0]
            graph.edge_index, _ = subgraph(train_node_idx, graph.edge_index) # remove val and test nodes (val and test are considered as isolated nodes)
            edge_index, _, mask = remove_isolated_nodes(graph.edge_index, num_nodes=graph.x.shape[0]) # remove all ioslated nodes and re-index nodes
            graph.x = graph.x[mask] 
        edge_index = to_undirected(graph.edge_index)
        edge_index = add_remaining_selfloop_for_isolated_nodes(edge_index, graph.x.shape[0])
        graph.edge_index = edge_index

        node_num = graph.x.shape[0]
        graph_num = min(self.graph_num, node_num)
        start_nodes = torch.randperm(node_num)[:graph_num]
        edge_index = graph.edge_index

        value = torch.arange(edge_index.size(1))
        self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(node_num, node_num)).t()

        view1_list = []
        view2_list = []

        views_cnt = 1 if self.aligned else 2
        for view_idx in range(views_cnt):
            current_nodes = start_nodes.clone()
            history = start_nodes.clone().unsqueeze(0)
            signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
            for i in range(self.walk_steps):
                seed = torch.rand([graph_num])
                nei = self.adj_t.sample(1, current_nodes).squeeze()
                sign = seed < self.restart_ratio
                nei[sign] = start_nodes[sign]
                history = torch.cat((history, nei.unsqueeze(0)), dim=0)
                signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
                current_nodes = nei
            history = history.T
            signs = signs.T
            
            for i in range(graph_num):
                path = history[i]
                sign = signs[i]
                node_idx = path.unique()
                sources = path[:-1].numpy().tolist()
                targets = path[1:].numpy().tolist()
                sub_edges = torch.IntTensor([sources, targets]).type_as(graph.edge_index)
                sub_edges = sub_edges.T[~sign[1:]].T
                # undirectional
                if sub_edges.shape[1] != 0:
                    sub_edges = to_undirected(sub_edges)
                view = self.adjust_idx(sub_edges, node_idx, graph, path[0].item())

                if self.aligned:
                    view1_list.append(view)
                    view2_list.append(copy.deepcopy(view))
                else:
                    if view_idx == 0:
                        view1_list.append(view)
                    else:
                        view2_list.append(view)
        return (view1_list, view2_list)

    def adjust_idx(self, edge_index, node_idx, full_g, center_idx):
        '''re-index the nodes and edge index

        In the subgraphs, some nodes are droppped. We need to change the node index in edge_index in order to corresponds 
        nodes' index to edge index
        '''
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edge_index[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edge_index[1].numpy().tolist()))

        edge_index = torch.IntTensor([sources_idx, target_idx]).type_as(full_g.edge_index)
        # x_view = Data(edge_index=edge_index, x=full_g.x[node_idx], center=node_idx_map[center_idx], original_idx=node_idx)
        x = obtain_attributes(Data(edge_index=edge_index), use_adj=True)
        x_view = Data(edge_index=edge_index, x=x, center=node_idx_map[center_idx], original_idx=node_idx, y=full_g.y[center_idx], root_n_index=node_idx_map[center_idx])
        return x_view
    
    
def collect_subgraphs(selected_id, graph, walk_steps=20, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_id # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
    
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T
    
    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).type_as(graph.edge_index)
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())

        graph_list.append(view)
    return graph_list
        
def adjust_idx(edge_index, node_idx, full_g, center_idx):
    '''re-index the nodes and edge index

    In the subgraphs, some nodes are droppped. We need to change the node index in edge_index in order to corresponds 
    nodes' index to edge index
    '''
    node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
    sources_idx = list(map(node_idx_map.get, edge_index[0].numpy().tolist()))
    target_idx = list(map(node_idx_map.get, edge_index[1].numpy().tolist()))

    edge_index = torch.IntTensor([sources_idx, target_idx]).type_as(full_g.edge_index)
    x_view = Data(edge_index=edge_index, x=full_g.x[node_idx], center=node_idx_map[center_idx], original_idx=node_idx, y=full_g.y[center_idx], root_n_index=node_idx_map[center_idx])
    return x_view

def ego_graphs_sampler(node_idx, data, hop=2):
    ego_graphs = []
    for idx in node_idx:
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, data.edge_index, relabel_nodes=True)
        # sub_edge_index = to_undirected(sub_edge_index)
        sub_x = data.x[subset]
        # center_idx = subset[mapping].item() # node idx in the original graph, use idx instead
        g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, y=data.y[idx], original_idx=idx) # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
        ego_graphs.append(g)
    return ego_graphs