import numpy as np
import torch
import torch.nn as nn

class ConvertToSequence(nn.Module):
    def __init__(self, num_return_buckets)