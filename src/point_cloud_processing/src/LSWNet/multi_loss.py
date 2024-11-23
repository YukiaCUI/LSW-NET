import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import random


def far_shuffle(tensor, dim=-1, min_distance=None):
    """
    Shuffles a tensor along a specified dimension, ensuring a minimum distance between original and shuffled positions.


    Args:
      tensor: The input PyTorch tensor.
      dim: The dimension along which to shuffle (default: -1, last dimension).
      min_distance: The minimum distance (in terms of index difference) between original and shuffled positions. If None, a default value is calculated, proportional to the size of dimension `dim`.

    Returns:
      A new tensor with the specified dimension shuffled.  Returns None if shuffling fails or if input is invalid.
    """

    if not isinstance(tensor, torch.Tensor):
        print("Error: Input must be a PyTorch tensor.")
        return None

    shape = list(tensor.shape)
    n = shape[dim]

    if min_distance is None:
        min_distance = max(1, n // 4)  #Default: Minimum distance is 1/4 the dimension size

    # Create an index mapping that ensures minimum distance.  This will require iteration in the worst case
    mapping = _create_distant_mapping(n, min_distance)

    if mapping is None:
      print("Could not generate distant mapping.")
      return None

    shuffled_tensor = tensor.clone()
    permutation = [slice(None)] * len(shape)
    permutation[dim] = torch.tensor(mapping)

    try:
      shuffled_tensor = shuffled_tensor[tuple(permutation)]

    except IndexError as e:
        print(f"Error during tensor reshaping/shuffling. Check dimensions. {e}")
        return None
    
    return shuffled_tensor

def _create_distant_mapping(n, min_distance):
    """
    Creates a permutation of indices [0, ..., n-1] ensuring a minimum distance.
    """
    mapping = list(range(n))
    shuffled_indices = []
    available_indices = list(range(n))

    for i in range(n):
      
      #Find candidate indices sufficiently far from existing shuffled positions.
      valid_candidates = [idx for idx in available_indices if min(abs(idx - x) for x in shuffled_indices) >= min_distance]
      
      if not valid_candidates:  #Fail to generate a valid mapping. 
          return None


      new_index = np.random.choice(valid_candidates) #randomly choose the farthest element to avoid bias
      shuffled_indices.append(new_index)
      available_indices.remove(new_index)


    return shuffled_indices

class MultiLoss(nn.Module):
    def __init__(self, alpha=0.5, lambda_reg=0.01):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.loss_spatem = 0
        self.loss_consist = 0
        self.l2_reg_loss = 0
        self.loss = 0
    
    def get_score(self, curve, tmse, alpha):
        # curve shape: (B, N)
        # tmse shape: (B, N)
        curve_score = curve / torch.max(curve, axis=1, keepdims=True)
        tmse_score = tmse / torch.max(tmse, axis=1, keepdims=True)
        
        return curve * alpha + tmse * (1 - alpha)

    def forward(self, features, curve, tmse):
        scores = self.get_score(curve, tmse, self.alpha) 
        weights = features[:, 2, :].squeeze()

        B, T, N = features.shape
        Ai = features[:, 2, :]
        Bi = (torch.sum(features, axis=1)-Ai) / (T-1)
        shuffled_features = far_shuffle(features, dim=2, min_distance=N//4)
        Aj = shuffled_features[:, 2, :]

        dpos = torch.abs(Ai - Bi)
        dneg = torch.abs(Ai - Aj)

        mse_loss = nn.MSELoss()
        self.loss_spatem = mse_loss(weights, scores)
        self.loss_consist = (dpos - dneg) * weights.mean()
        self.loss = self.loss_spatem + self.loss_consist
        
        # L2 Regularization on features
        # Compute the L2 norm of the features tensor (excluding batch dimension)
        features_flattened = features.view(-1, N)  # Flatten the tensor (B*T, N) for norm calculation
        self.l2_reg_loss = torch.norm(features_flattened, p=2)  # L2 regularization (sum of squares)

        # Apply regularization
        self.loss += self.lambda_reg * self.l2_reg_loss  # Add L2 regularization

        return self.loss
         


