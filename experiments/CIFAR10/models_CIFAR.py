# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import collections

import tensorflow as tf

from deel.lipdp.layers import DP_AddBias
from deel.lipdp.layers import DP_BoundedInput
from deel.lipdp.layers import DP_ClipGradient
from deel.lipdp.layers import DP_Flatten
from deel.lipdp.layers import DP_GroupSort
from deel.lipdp.layers import DP_Lambda
from deel.lipdp.layers import DP_LayerCentering
from deel.lipdp.layers import DP_Permute
from deel.lipdp.layers import DP_QuickSpectralDense
from deel.lipdp.layers import DP_Reshape
from deel.lipdp.layers import DP_ScaledGlobalL2NormPooling2D
from deel.lipdp.layers import DP_ScaledL2NormPooling2D
from deel.lipdp.layers import DP_SpectralConv2D
from deel.lipdp.layers import DP_SpectralDense
from deel.lipdp.layers import make_residuals
from deel.lipdp.model import DP_Model
from deel.lipdp.model import DP_Sequential


def create_MLP_Mixer(dp_parameters, dataset_metadata, cfg, upper_bound):
    """Creates a MLP-Mixer network.

    The cfg object must contain some information:
        - cfg.add_biases (bool): DP_AddBias layers after each linear layer.
        - cfg.layer_centering (bool): DP_LayerCentering layers after each activation.
        - cfg.skip_connections (bool): skip connections in the MLP-Mixer network.
        - cfg.num_mixer_layers (int): number of mixer layers.
        - cfg.patch_size (int): size of the patches.
        - cfg.hidden_size (int): size of the hidden layer.
        - cfg.mlp_seq_dim (int): size of the hidden layer in the MLP block.
        - cfg.mlp_channel_dim (int): size of the hidden layer in the channel block.
        - cfg.clip_loss_gradient (float): clip the gradient of the loss to this value.

    Args:
        dp_parameters: parameters for differentially private training
        dataset_metadata: metadata of the dataset, for privacy accounting
        cfg: configuration containing information for DP_Sequential and MLP-Mixer
            hyper-parameters
        upper_bound (float): maximum norm of the input (clipped if input norm is higher)

    Returns:
        DP_Sequential: DP MLP-Mixer network
    """
    input_shape = dataset_metadata.input_shape
    layers = [DP_BoundedInput(input_shape=input_shape, upper_bound=upper_bound)]

    layers.append(
        DP_Lambda(
            tf.image.extract_patches,
            arguments=dict(
                sizes=[1, cfg.patch_size, cfg.patch_size, 1],
                strides=[1, cfg.patch_size, cfg.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            ),
        )
    )

    # layers.append(DP_SpectralConv2D(filters=hidden_dim, kernel_size=patch_size, use_bias=False, strides=patch_size, padding="same"))
    seq_len = (input_shape[0] // cfg.patch_size) * (input_shape[1] // cfg.patch_size)

    layers.append(DP_Reshape((seq_len, (cfg.patch_size**2) * input_shape[-1])))
    layers.append(
        DP_QuickSpectralDense(
            units=cfg.hidden_size, use_bias=False, kernel_initializer="identity"
        )
    )

    for _ in range(cfg.num_mixer_layers):
        to_add = [
            DP_Permute((2, 1)),
            DP_QuickSpectralDense(
                units=cfg.mlp_seq_dim, use_bias=False, kernel_initializer="identity"
            ),
        ]
        if cfg.add_biases:
            to_add.append(DP_AddBias())
        to_add.append(DP_GroupSort(2))
        if cfg.layer_centering:
            to_add.append(DP_LayerCentering())
        to_add += [
            DP_QuickSpectralDense(
                units=seq_len, use_bias=False, kernel_initializer="identity"
            ),
            DP_Permute((2, 1)),
        ]

        if cfg.skip_connections:
            layers += make_residuals("1-lip-add", to_add)
        else:
            layers += to_add

        to_add = [
            DP_QuickSpectralDense(
                units=cfg.mlp_channel_dim, use_bias=False, kernel_initializer="identity"
            ),
        ]
        if cfg.add_biases:
            to_add.append(DP_AddBias())
        to_add.append(DP_GroupSort(2))
        if cfg.layer_centering:
            to_add.append(DP_LayerCentering())
        to_add.append(
            DP_QuickSpectralDense(
                units=cfg.hidden_size, use_bias=False, kernel_initializer="identity"
            )
        )

        if cfg.skip_connections:
            layers += make_residuals("1-lip-add", to_add)
        else:
            layers += to_add

    # TO REPLACE ?
    # layers.append(DP_LayerCentering())
    layers.append(DP_Flatten())
    layers.append(
        DP_QuickSpectralDense(units=10, use_bias=False, kernel_initializer="identity")
    )
    if cfg.clip_loss_gradient is not None:
        layers.append(DP_ClipGradient(cfg.clip_loss_gradient))

    model = DP_Model(
        layers,
        dp_parameters=dp_parameters,
        dataset_metadata=dataset_metadata,
        name="mlp_mixer",
    )

    model.build(input_shape=(None, *input_shape))

    return model


def create_VGG(dp_parameters, dataset_metadata, cfg, upper_bound):
    depth_conv = dict(VGG5_small=1, VGG5_large=1, VGG8_small=2, VGG8_large=2)
    init_filters = dict(VGG5_small=32, VGG5_large=64, VGG8_small=32, VGG8_large=64)

    assert cfg.architecture in depth_conv.keys()
    return VGG_factory(
        dp_parameters,
        dataset_metadata,
        cfg,
        depth_conv=depth_conv[cfg.architecture],
        depth_fc=1,
        init_filters=init_filters[cfg.architecture],
        upper_bound=upper_bound,
    )


def VGG_factory(
    dp_parameters,
    dataset_metadata,
    cfg,
    depth_conv,
    depth_fc,
    init_filters,
    upper_bound,
):
    """Creates a VGG-like network.

    The cfg object must contain some information:
        - cfg.add_biases (bool): DP_AddBias layers after each linear layer.
        - cfg.layer_centering (bool): DP_LayerCentering layers after each activation.

    The VGG network is composed of three blocks of `depth` convolutional layers. A
    pooling layer is appended to each block. The network ends with `depth_fc + 1`
    fully-connected layers (the last one is the classification layer). The width of each
    block is:
        - filters in convolutional block 1: init_filters
        - filters in convolutional block 2: init_filters * 2
        - filters in convolutional block 3: init_filters * 4
        - units in fully-connected layers: init_filters * 8

    Args:
        cfg: configuration containing information for DP_Sequential and VGG
            hyper-parameters
        depth_conv (int): number of convolutions in the three convolutional blocks.
            Usually depth_conv=1 or 2.
        depth_fc (int): number of fully-connected layers before the classification
            layer. Usually depth_fc=1 or 2.
        init_filters (int): number of filters in the first convolution block. Usually a
            power of 2 (e.g. 32, 64 or 128).
        upper_bound (float): maximum norm of the input (clipped if input norm is higher)

    Returns:
        DP_Sequential: DP VGG network
    """
    layers = []
    layers.append(DP_BoundedInput(input_shape=(32, 32, 3), upper_bound=upper_bound))

    # Convolutional block 1
    for _ in range(depth_conv):
        layers.append(
            DP_SpectralConv2D(
                filters=init_filters,
                kernel_size=5,
                kernel_initializer="orthogonal",
                strides=1,
                use_bias=False,
            )
        )
        layers.append(DP_AddBias(norm_max=1))
        layers.append(DP_GroupSort(2))
        layers.append(DP_LayerCentering())
    layers.append(DP_ScaledL2NormPooling2D(pool_size=2, strides=2))

    # Convolutional block 2
    for _ in range(depth_conv):
        layers.append(
            DP_SpectralConv2D(
                filters=init_filters * 2,
                kernel_size=3,
                kernel_initializer="orthogonal",
                strides=1,
                use_bias=False,
            )
        )
        layers.append(DP_AddBias(norm_max=1))
        layers.append(DP_GroupSort(2))
        layers.append(DP_LayerCentering())
    layers.append(DP_ScaledL2NormPooling2D(pool_size=2, strides=2))

    # Convolutional block 3
    for _ in range(depth_conv):
        layers.append(
            DP_SpectralConv2D(
                filters=init_filters * 4,
                kernel_size=3,
                kernel_initializer="orthogonal",
                strides=1,
                use_bias=False,
            )
        )
        layers.append(DP_AddBias(norm_max=1))
        layers.append(DP_GroupSort(2))
        layers.append(DP_LayerCentering())
    layers.append(DP_ScaledGlobalL2NormPooling2D())

    # Fully connected layers
    for _ in range(depth_fc):
        layers.append(
            DP_SpectralDense(
                init_filters * 8, use_bias=False, kernel_initializer="orthogonal"
            )
        )
        layers.append(DP_AddBias(norm_max=1))
        layers.append(DP_GroupSort(2))
        layers.append(DP_LayerCentering())

    layers.append(DP_SpectralDense(10, use_bias=False, kernel_initializer="orthogonal"))
    layers.append(DP_AddBias(norm_max=1))
    layers.append(DP_ClipGradient(cfg.clip_loss_gradient))

    # Remove DP_AddBias and DP_LayerCentering layers if required
    if cfg.add_biases is False:
        layers = [layer for layer in layers if not isinstance(layer, DP_AddBias)]
    if cfg.layer_centering is False:
        layers = [layer for layer in layers if not isinstance(layer, DP_LayerCentering)]

    model = DP_Sequential(
        layers,
        dp_parameters=dp_parameters,
        dataset_metadata=dataset_metadata,
    )
    return model


# -------------------------------------------------------------------------
# ResNet
# -------------------------------------------------------------------------
ModelParams = collections.namedtuple("ModelParams", ["repetitions", "init_filters"])

RESNET_MODELS_DICT = {
    "resnet6a_small": ModelParams((2,), 32),
    "resnet6a_large": ModelParams((2,), 64),
    "resnet6b_small": ModelParams(
        (
            1,
            1,
        ),
        32,
    ),
    "resnet6b_large": ModelParams(
        (
            1,
            1,
        ),
        32,
    ),
    "resnet8_small": ModelParams((1, 1, 1), 32),
    "resnet8_large": ModelParams((1, 1, 1), 64),
    "resnet10_small": ModelParams((2, 2), 32),
    "resnet10_large": ModelParams((2, 2), 64),
}


# Helper function
def handle_block_names(stage, block):
    name_base = "stage{}_unit{}_".format(stage + 1, block + 1)
    conv_name = name_base + "spconv"
    lc_name = name_base + "lc"
    gs_name = name_base + "gs"
    pool_name = name_base + "pool"
    return conv_name, lc_name, gs_name, pool_name


# Residual block
def residual_conv_block(filters, stage, block):
    # get params and names of layers
    conv_name, lc_name, gs_name, pool_name = handle_block_names(stage, block)

    layers = []
    to_add = []

    # first block: pool (except block 0) + additional conv (-> filters)
    if block == 0:
        if stage != 0:
            layers += [
                DP_ScaledL2NormPooling2D(pool_size=2, strides=2, name=pool_name + "1")
            ]

        layers += [
            DP_SpectralConv2D(
                filters=filters,
                kernel_size=(1, 1),
                kernel_initializer="orthogonal",
                strides=(1, 1),
                use_bias=False,
                name=conv_name + "0",
            )
        ]

    # continue with convolution layers
    to_add += [
        DP_LayerCentering(name=lc_name + "1"),
        DP_GroupSort(2, name=gs_name + "1"),
        DP_SpectralConv2D(
            filters=filters,
            kernel_size=(3, 3),
            kernel_initializer="orthogonal",
            strides=(1, 1),
            use_bias=False,
            name=conv_name + "1",
        ),
        DP_LayerCentering(name=lc_name + "2"),
        DP_GroupSort(2, name=gs_name + "2"),
        DP_SpectralConv2D(
            filters=filters,
            kernel_size=(3, 3),
            kernel_initializer="orthogonal",
            strides=(1, 1),
            use_bias=False,
            name=conv_name + "2",
        ),
    ]

    # add residual connection
    layers += make_residuals("1-lip-add", to_add)
    return layers


# ResNet Builder
def create_ResNet(dp_parameters, dataset_metadata, cfg, upper_bound):
    model_params = RESNET_MODELS_DICT[cfg.architecture]

    # CIFAR10
    classes = 10
    input_shape = (32, 32, 3)
    layers = [
        DP_BoundedInput(input_shape=input_shape, name="data", upper_bound=upper_bound)
    ]

    # get parameters for model layers
    init_filters = model_params.init_filters

    # resnet bottom
    layers += [
        # set stride 2 in pooling instead of conv
        DP_SpectralConv2D(
            filters=init_filters,
            kernel_size=(3, 3),
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
            name="conv0",
        ),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2, name="pool0"),
        DP_LayerCentering(name="lc0"),
        DP_GroupSort(2, name="gs0"),
    ]

    # resnet body
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):
            filters = init_filters * (2**stage)
            layers += residual_conv_block(filters, stage, block)

    # resnet top
    layers += [
        DP_ScaledGlobalL2NormPooling2D(name="globalpool1"),
        DP_SpectralDense(classes, use_bias=False, name="fc1"),
        DP_ClipGradient(cfg.clip_loss_gradient, name="clipgrad"),
    ]

    model = DP_Model(
        layers,
        cfg=cfg,
        dp_parameters=dp_parameters,
        dataset_metadata=dataset_metadata,
        noisify_strategy=cfg.noisify_strategy,
        name=cfg.architecture,
    )
    model.build(input_shape=(None, *input_shape))
    return model
