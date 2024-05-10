# probabilistic-pruning
Probabilistic Pruning of DNNs

## How To Use
Our three main files containing code are `resnet_optimized.ipynb`, `EfficientNet_optimized.ipynb`, and `pruning_funcs.py`.

### Main Files
`resnet_optimized.ipynb`, `EfficientNet_optimized.ipynb`

These files are written to train/test the ResNet and EfficientNet models, respectively.

Each file contains cells that define the models, train the models, and test various pruning methods for sparsity and accuracy.

These two files are the main place to look for experiments and experimental results. We have stored our output in the notebooks, with the same values and figures that appear in our reports. Note that some of our experiments are inherently non-deterministic, so while some experiments will have *similar* numbers, we do not expect these experiments to give the *exact same* numbers every time.

### Stored State
`resnet_state`, `efficentnet_state`

These binary files store the trained weights for the ResNet and EfficientNet models, respectively.

### Supplementary
`pruning_funcs.py`

This file contains all of our developed methods for pruning the torchvision models in the above files.

This file also contains methods for evaluating the number/percentage of pruned weights in a model.

### Deprecated Files
`working_trained.ipynb`, `state`

Contains old work.

## Required Packages

`torch`, `torchvision`, `numpy`, `matplotlib`, `timm`
