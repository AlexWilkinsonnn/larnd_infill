import torch

def mse_gap_weighted(s_pred, s_target):
    """Loss function with different weighting at for gaps. Assumes a signal mask has been used"""
    target_feats_padded = s_target.features_at_coordinates(s_pred.C.type(torch.float32))
    pass
