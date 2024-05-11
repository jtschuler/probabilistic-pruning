# Probabilistic Pruning of Neural Networks

## Overview
This repository contains the resources for implementing probabilistic pruning on Deep Neural Networks (DNNs). The main notebooks, `resnet_optimized.ipynb` and `EfficientNet_optimized.ipynb`, along with the supporting script `pruning_funcs.py`, are detailed below.

### Primary Resources
`resnet_optimized.ipynb` and `EfficientNet_optimized.ipynb`:

These notebooks are dedicated to training and testing the ResNet and EfficientNet models. They include comprehensive instructions for model definition, training procedures, and the application of various pruning techniques to assess sparsity and accuracy. These notebooks serve as the primary sources for experimental analysis and contain outputs that are integral to our reports. Due to the probabilistic nature of the experiments, results may vary slightly in each run, reflecting the stochastic nature of the process.

### State Files
`resnet_state`, `efficientnet_state`:

These binary files store the trained model weights for the ResNet and EfficientNet models, enabling reproducibility and further experimentation without the need to retrain from scratch.

### Utility Script
`pruning_funcs.py`:

This script encompasses all the core logic functions necessary for pruning models documented in the notebooks above. 

### Legacy Files
`deprecated_working_trained.ipynb`, `state`:

These files contain earlier iterations of our work and are maintained for historical reference.

## Dependencies
To run the notebooks and scripts provided in this repository, the following packages are required:
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `timm`

This setup guide aims to facilitate a structured approach to exploring probabilistic pruning within DNNs, ensuring clarity and consistency in replication and experimentation.

