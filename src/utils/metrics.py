import torch
import numpy as np
import os

def masked_mse(preds, labels, null_val = np.nan, mask = None):
    '''
    Calculate MSE.
    The missing values in labels will be masked.
    '''
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val = np.nan, mask = None):
    if mask == None:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    else:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, mask = mask))


def masked_mae(preds, labels, null_val = np.nan, mask = None):
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(pred, real, null_value =np.nan):
    mae = masked_mae(pred, real, null_value).item()
    rmse = masked_rmse(pred, real, null_value).item()
    return mae, rmse


def sudden_changes_mask(labels, datapath, null_val = np.nan, threshold_start = 75, threshold_change = 20):
    '''
    Create the mask for sudden change case.
    The parameter 'threshold_start' and 'threshold_change' can be changed.
    '''
    path = os.path.join(datapath, 'mask_sudden_change_{}_{}.pth'.format(threshold_start, threshold_change))
    if os.path.exists(path):
        mask = torch.load(path)
    else:
        labels = labels.squeeze(-1)
        b, t, n= labels.shape
        mask = torch.zeros(size = (b, t, n))
        mask_ones = torch.ones(size = (b, n))
        mask_zeros = torch.zeros(size = (b, n))
        for t in range(1, 24):
            prev = labels[:, t-1]
            curr = labels[:, t]
            mask[:, t] = torch.where((torch.BoolTensor(curr > threshold_start)), mask_ones, mask[:, t]) 
            mask[:, t] = torch.where(torch.abs(torch.Tensor(curr - prev))> threshold_change, mask_ones, mask[:, t])
            if not np.isnan(null_val):
                mask[:, t ] = torch.where(torch.BoolTensor(prev < null_val + 0.1), mask_zeros, mask[:, t ])
            else:
                mask[:, t ] = torch.where(torch.isnan(curr), mask_zeros, mask[:, t ])
        mask = mask.unsqueeze(-1)
        torch.save(mask, path)
    return mask

def compute_sudden_change(mask, pred, real, null_value):
    mae = masked_mae(pred, real, null_value, mask).item()
    rmse = masked_rmse(pred, real, null_value, mask).item()
    return mae, rmse

