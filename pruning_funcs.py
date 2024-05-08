import numpy as np
from numpy import random
import torch
import copy

# Eval metric
# From https://discuss.pytorch.org/t/how-to-count-the-number-of-zero-weights-in-a-pytorch-model/13549/2
def count_zero_weights(model):
    zeros = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            zeros += param.numel() - param.nonzero().size(0)
    return zeros

def percent_zero_weights(model):
    num_zeros = count_zero_weights(model)
    total_num_weights = sum(p.numel() for name, p in model.named_parameters() if 'weight' in name)
    return num_zeros / total_num_weights * 100

# Laplace-based pruning methods

def laplace_pdf(x: torch.Tensor, scale, loc=0.):
    return torch.exp(-abs(x-loc)/scale)/(2.*scale)

def normalized_laplace_pdf(x: torch.Tensor, loc=0., scale=0.5):
    laplace_at = lambda w : np.exp(-abs(w-loc)/scale)/(2.*scale)
    return (laplace_pdf(x, scale) - laplace_at(1)) / (laplace_at(loc) - laplace_at(1))
     
def laplace_prune(model, device, scale=0.5):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                laplace_tensor = laplace_pdf(param, scale=scale)
                mask = torch.bernoulli(1 - laplace_tensor).to(device)
                param.mul_(mask)
     
def normalized_laplace_prune(model, device, scale=0.5):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                laplace_tensor = normalized_laplace_pdf(param, scale=scale)
                mask = torch.bernoulli(1 - laplace_tensor).to(device)
                param.mul_(mask)

# Percent-based pruning methods

def bernoulli_prune(model, device, probability=0.99):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = torch.bernoulli(torch.full(tuple(param.shape), probability)).to(device)
                param.mul_(tensor)


def percent_prune(model, device, percent=1.5, debug=False):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                num_elems = 1
                for val in param.shape:
                    num_elems = num_elems * val
                flattened = torch.sort(torch.flatten(torch.abs(param)))

                if debug:
                    print(torch.abs(param))
                    print(torch.min(torch.abs(param)))
                    print(flattened)

                prune_cutoff_idx = int(num_elems*percent/100)
                prune_cutoff_val = flattened[0][prune_cutoff_idx]
                mask = torch.abs(param) >= prune_cutoff_val

                if debug:
                    print(mask)

                param.mul_(mask)

def percent_prune_with_bernoulli(model, device, percent=5, p_success=0.5, debug=False):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                num_elems = 1
                for val in param.shape:
                    num_elems = num_elems * val
                flattened = torch.sort(torch.flatten(torch.abs(param)))

                if debug:
                    print(torch.abs(param))
                    print(torch.min(torch.abs(param)))
                    print(flattened)

                prune_cutoff_idx = int(num_elems*percent/100)
                prune_cutoff_val = flattened[0][prune_cutoff_idx]
                mask = torch.abs(param) >= prune_cutoff_val


                parm_copy = copy.deepcopy(param)
                inv_mask = torch.abs(param) < prune_cutoff_val
                parm_copy.mul_(inv_mask)

                tensor = torch.bernoulli(torch.full(tuple(param.shape), p_success)).to(device)
                parm_copy.mul_(tensor)



                if debug:
                    print(mask)

                param.mul_(mask)
                param += parm_copy




def percent_prune_min_max(model, device, percent=1, debug=False):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                num_elems = 1
                for val in param.shape:
                    num_elems = num_elems * val
                flattened = torch.sort(torch.flatten(torch.abs(param)))

                if debug:
                    print(torch.abs(param))
                    print(torch.min(torch.abs(param)))
                    print(flattened)

                prune_cutoff_idx = int(num_elems*percent/100)
                prune_cutoff_val_min = flattened[0][prune_cutoff_idx]


                prune_cutoff_idx = int(num_elems*(1-percent/100))
                prune_cutoff_val_max = flattened[0][prune_cutoff_idx]


                
                mask = torch.abs(param) >= prune_cutoff_val_min
                mask = torch.abs(param) <= prune_cutoff_val_max

                if debug:
                    print(mask)

                param.mul_(mask)