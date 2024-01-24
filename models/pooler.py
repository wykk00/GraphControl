import torch
import torch_scatter


def subg_pooling(reps, data):
    batch_size = data.batch.max().cpu().item() + 1
    graphsize_perbat = torch.zeros(batch_size, dtype=data.batch.dtype, device=data.batch.device)
    tmp = torch.ones_like(data.batch)
    torch_scatter.scatter_add(tmp, data.batch, out=graphsize_perbat)
    center_indices = data.center
    center_mask = torch.zeros_like(data.batch)

    pointer = 0
    for i in range(0, batch_size):
        center_mask[center_indices[i] + pointer] = 1
        pointer += graphsize_perbat[i]
    
    center_mask = center_mask.bool()
    return reps[center_mask], data.y