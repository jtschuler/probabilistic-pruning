import numpy as np
from numpy import random
import torch
import copy

# Sparsity Evaluation Metrics
# From https://discuss.pytorch.org/t/how-to-count-the-number-of-zero-weights-in-a-pytorch-model/13549/2
def count_zero_weights(model):
    '''Count the number of model weights that are zero.'''
    zeros = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            zeros += param.numel() - param.nonzero().size(0)
    return zeros
def percent_zero_weights(model):
    '''Compute the percent of model weights that are zero.'''
    num_zeros = count_zero_weights(model)
    total_num_weights = sum(p.numel() for name, p in model.named_parameters() if 'weight' in name)
    return num_zeros / total_num_weights * 100



################################################################# Algorithm 4 #################################################################

# Baseline Percent Pruning: Algorithm 4 in submitted paper
def bottom_percent_prune(model, device, percent=1.5):
    '''Prune the bottom 𝑥% of weightsacross the model'''
    with torch.no_grad():
        all_weights = torch.Tensor().to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                flattened = torch.flatten(torch.abs(param))
                all_weights = torch.cat((all_weights, flattened))

        num_elems = all_weights.numel()
        all_weights = torch.sort(all_weights)

        prune_cutoff_idx = int(num_elems*percent/100)
        prune_cutoff_val = all_weights[0][prune_cutoff_idx]
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param) > prune_cutoff_val
                param.mul_(mask)

#############################################################################################################################################



################################################################# Algorithm 1 #################################################################

# Percent Pruning by Layer: Algorithm 1 in submitted paper
def percent_prune_by_layer(model, device, percent=1.5):
    '''prune the bottom 𝑋 % of weights within each layer'''
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                num_elems = 1
                for val in param.shape:
                    num_elems = num_elems * val
                flattened = torch.sort(torch.flatten(torch.abs(param)))
                prune_cutoff_idx = int(num_elems*percent/100)
                prune_cutoff_val = flattened[0][prune_cutoff_idx]
                mask = torch.abs(param) >= prune_cutoff_val
                param.mul_(mask)

#############################################################################################################################################




################################################################# Algorithm 2 #################################################################

# Stochastic Percent Pruning: Algorithm 2 in submitted paper
def percent_prune_with_bernoulli(model, device, percent=5, p_success=0.5, debug=False):
    '''prune each weight in bottom 𝑋 𝑓 % of weights, with 1/𝑓 probability'''
    with torch.no_grad():
        all_weights = torch.Tensor().to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                flattened = torch.flatten(torch.abs(param))
                all_weights = torch.cat((all_weights, flattened))

        num_elems = all_weights.numel()
        all_weights = torch.sort(all_weights)

        prune_cutoff_idx = int(num_elems*percent/100)
        prune_cutoff_val = all_weights[0][prune_cutoff_idx]
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param) > prune_cutoff_val

                param_copy = copy.deepcopy(param)
                inv_mask = torch.abs(param) < prune_cutoff_val
                param_copy.mul_(inv_mask)

                tensor = torch.bernoulli(torch.full(tuple(param.shape), p_success)).to(device)
                param_copy.mul_(tensor)

                param.mul_(mask)
                param += param_copy

#############################################################################################################################################



################################################################# Algorithm 3 #################################################################

# Laplace Pruning: Algorithm 3 in Paper
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

#############################################################################################################################################




################################################################# Experimental Unused #################################################################
def bernoulli_prune(model, device, probability=0.99):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = torch.bernoulli(torch.full(tuple(param.shape), probability)).to(device)
                param.mul_(tensor)

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

# Threshold Pruning Functions
def threshold_prune(model, threshold):
    '''Prune all weights less than a threshold t'''
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param) > threshold
                param.mul_(mask)
#############################################################################################################################################