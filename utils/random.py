import random

import numpy as np
import torch


def reset_random_seed(seed):
    r"""
    Initial process for fixing all possible random seed.

    Args:
       config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.random_seed`)
    """
    # Fix Random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Default state is a training state
    torch.enable_grad()