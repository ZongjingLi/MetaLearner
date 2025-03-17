import torch
import torch.nn as nn

def weight_sum(values, weights):
    """
    weight sum of 'values' and 'weights'
    values: [B,N,D] weights: [B,N,M]
    """
    if len(values.shape) == 3:
        outputs = torch.bmm(weights.permute(0,2,1),values)
    if len(values.shape) == 2:
        outputs = torch.mm(weights.permute(0,2,1),values)
    return outputs

def weight_mean(values, weights):
    if len(values.shape) == 3:
        outputs = torch.bmm(weights.permute(0,2,1),values)
    if len(values.shape) == 2:
        outputs = torch.mm(weights.permute(0,2,1),values)
    return outputs

def mask_entropy(masks):
    """
    entropy regularization for mask specifically
    masks: BxNxM -> Bx1
    """
    return masks