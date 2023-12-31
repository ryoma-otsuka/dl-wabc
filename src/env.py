import random
import numpy as np
import torch


def reset_seed(seed: int = 0):
    """Reset random seed (random, numpy, torch-cpu, torch-cuda) for reproducibility.
    Args:
        seed (int): random seed. (Default: 1)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
def reset_seed_ml(seed: int = 0):
    """Reset random seed (random, numpy, torch-cpu, torch-cuda) for reproducibility.
    Args:
        seed (int): random seed. (Default: 1)
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
