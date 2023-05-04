# CIFAR-10 experiments

## Network architectures

Different network architectures are proposed to train a DP model: VGG, ResNet and
MLPMixer.

### VGG architectures

Four VGG models are proposed with a variety of depths and widths. Only four are provided
to limit the grid search possibilities in W&B sweeps. But it is possible to propose new
VGG models based on the `VGG_factory` function. The depth and width of the network are
displayed in the model's name:

- Depth is characterized by the number after VGG: VGG5 contains 5 linear layers (3
  convs + 2 FC), VGG8 contains 8 layers (6 convs + 2 FC).
- Width is defined by "small" (starts with 32 filters) or "large" (starts with 64
  filters).

Here is a short description of the four models:

| Network    | Num. params | Max batch size\* | Runtime\*  |
| ---------- | ----------- | ---------------- | ---------- |
| VGG5_small | 260K        | 9000             | 2s / epoch |
| VGG8_small | 680K        | 7000             | 4s / epoch |
| VGG5_large | 1.0M        | 4000             | 4s / epoch |
| VGG8_large | 2.7M        | 3000             | 8s / epoch |

\*Measures were performed on a NVIDIA GeForce RTX 3080 and are approximate. "Max batch
size" means that for larger batch sizes, the run crashes with "out of memory" error.

### ResNet architectures

A few ResNet-like models are proposed with a variety of depths and widths. It is
possible to propose new configurations by editing the `RESNET_MODELS_DICT` dictionary.
The depth and width of the network are displayed in the model's name:

- Depth is characterized by the number after resnet: resnet6 contains 6 layers (5
  convs + 1 FC), resnet8 contains 8 layers (7 convs + 1 FC).
- Width is defined by "small" (starts with 32 filters) or "large" (starts with 64
  filters).
- Note that for a fixed number of layers, there can be multiple architecture variations,
depending on the number of repetitions of residual blocks (containing 2 convs) between
two pooling layers. Here resnet6b refers to a ResNet6 with two pooling layers in its body while
resnet6a has only one.

<!-- TODO: Update when machines available

Here is a short description of the proposed models:


| Network            | Num. params | Max batch size\* | Runtime\*  |
| ------------------ | ----------- | ---------------- | ---------- |
| resnet6a_small     | 000M        | 0000             | 0s / epoch |
| resnet6a_large     | 000M        | 0000             | 0s / epoch |
| resnet6b_small     | 000M        | 0000             | 0s / epoch |
| resnet6b_large     | 000M        | 0000             | 0s / epoch |
| resnet8_small      | 000M        | 0000             | 0s / epoch |
| resnet8_large      | 000M        | 0000             | 0s / epoch |
| resnet10_small     | 000M        | 0000             | 0s / epoch |
| resnet10_large     | 000M        | 0000             | 0s / epoch |

\*Measures were performed on a NVIDIA GeForce RTX 3080 and are approximate. "Max batch
size" means that for larger batch sizes, the run crashes with "out of memory" error. -->
