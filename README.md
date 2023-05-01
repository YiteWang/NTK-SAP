# NTK-SAP: Improving neural network pruning by aligning training dynamics

Yite Wang, Dawei Li, Ruoyu Sun

In ICLR 2023.

## Overview

This is the PyTorch implementation of NTK-SAP: Improving neural network pruning by aligning training dynamics.

## Installation

To run our code, then install all dependencies
```
pip install -r requirements.txt
```
## Running
Below is a description of the major sections of the code base. Run `python main.py --help` for a complete description of flags and hyperparameters.

### 1. Prepare the datasets
MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet will be downloaded automatically. For ImageNet experiment, please download it to `Data/imagenet_raw/`, or change corresponding path in `Utils/load.py`.

### 2. Run foresight pruning experiments

Note experiments of ImageNet requires running code to prune and train separately, see the argument `experiment`. For other experiments, models will be trained right after pruning. We include a few important arguments:

- `--experiment`: For CIFAR-10, CIFAR-100, and Tiny-ImageNet experiments, you can either use `singleshot` or `multishot`. For ImageNet experiment, please use `multishot_ddp_prune` to get mask then train with `multishot_ddp_train`.
- `--dataset`: Which dataset to use, to reproduce our results, use `cifar10`, `cifar100`, `tiny-imagenet`, and `imagenet`.
- `--model-class`: For CIFAR-10 and CIFAR-100 experiments, please use `lottery`. For Tiny-imagenet and ImageNet experiments, please use `imagenet`.
- `--model`:  Which model architecture to use. In our experiments, we use `resnet20`, `vgg16-bn`, `resnet18`, and `resnet50`.
- `--pruner`: Which pruning algorithms to use, choose from: `rand`, `mag`, `snip`, `grasp`, `synflow`, `itersnip`, `NTKSAP`.
- `--prune-batch-size`: Batch size of pruning datasets.
- `--compression`: You can use this argument to change sparsity for `singleshot` experiments. Specifically, the target density will be $0.8^{\text{compression}}$. For `multishot` experiments, please refer to `--compression-list`.
- `--prune-train-mode`: Set this to `True` if you use pruning algorithms except Synflow.
- `--prune-epochs`: Number of pruning iterations $T$.
- `--ntksap_R`: Number of resampling procedures, only change this for CIFAR-10 experiment.
- `--ntk_epsilon`: Perturbation hyper-parameter used in NTK-SAP.

A sample script can be found in `scripts/run.sh`.

## Acknowledgement
Our code is developed based on the [Synflow](https://arxiv.org/abs/2006.05467) code: https://github.com/ganguli-lab/Synaptic-Flow.
