import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def run_blocks(blocks, x, use_checkpoint=False):
    """
    Apply a sequence of blocks to x in sequence.
    When use_checkpoint is True, uses torch.utils.checkpoint.checkpoint 
    with use_reentrant=False to save memory by recomputing activations during backward.
    """
    for block in blocks:
        if use_checkpoint:
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x