import numpy as np
from numpy import random
import torch
import copy

# def laplace_pdf(x: torch.Tensor, loc=0, scale=1):
#     y = np.arange(-2, 2, 0.1)
#     pdf = np.exp(-abs(y-loc)/scale)/(2.*scale)
#     import matplotlib.pyplot as plt
#     plt.plot(y, pdf)

#     return torch.exp(-abs(x-loc)/scale)/(2.*scale)
     
# def laplace_prune(model, device):
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if 'weight' in name:
#                 laplace_tensor = laplace_pdf(param)
#                 print(param)
#                 print(laplace_tensor)
#                 print(1 - laplace_tensor)
#                 mask = torch.bernoulli(1 - laplace_tensor).to(device)
#                 print(mask)
#                 break
#                 # param.mul_(mask)


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

def percent_prune_with_bernulli(model, device, percent=5, debug=False):
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

                tensor = torch.bernoulli(torch.full(tuple(param.shape), 0.5)).to(device)
                parm_copy.mul_(tensor)



                if debug:
                    print(mask)

                param.mul_(mask)
                param += parm_copy