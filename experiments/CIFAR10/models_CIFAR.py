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
import tensorflow as tf

from deel.lipdp.layers import DP_AddBias
from deel.lipdp.layers import DP_BoundedInput
from deel.lipdp.layers import DP_ClipGradient
from deel.lipdp.layers import DP_Flatten
from deel.lipdp.layers import DP_GroupSort
from deel.lipdp.layers import DP_Lambda
from deel.lipdp.layers import DP_LayerCentering
from deel.lipdp.layers import DP_Permute
from deel.lipdp.layers import DP_Reshape
from deel.lipdp.layers import DP_ScaledL2NormPooling2D
from deel.lipdp.layers import DP_SpectralConv2D
from deel.lipdp.layers import DP_SpectralDense
from deel.lipdp.layers import make_residuals
from deel.lipdp.model import DP_Model
from deel.lipdp.model import DP_Sequential


def create_MLP_Mixer(cfg, upper_bound):
    input_shape = (32, 32, 3)
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
    layers.append(DP_SpectralDense(units=cfg.hidden_size, use_bias=False))

    for _ in range(cfg.num_mixer_layers):
        to_add = [
            DP_Permute((2, 1)),
            DP_SpectralDense(units=cfg.mlp_seq_dim, use_bias=False),
            DP_LayerCentering(),
            DP_GroupSort(2),
            DP_SpectralDense(units=seq_len, use_bias=False),
            DP_Permute((2, 1)),
        ]

        if cfg.skip_connections:
            layers += make_residuals("1-lip-add", to_add)
        else:
            layers += to_add

        to_add = [
            # channel mixing
            # TODO: add LayerNorm ?
            DP_SpectralDense(units=cfg.mlp_channel_dim, use_bias=False),
            DP_GroupSort(2),
            DP_SpectralDense(units=cfg.hidden_size, use_bias=False),
        ]

        if cfg.skip_connections:
            layers += make_residuals("1-lip-add", to_add)
        else:
            layers += to_add

    # TO REPLACE ?
    # layers.append(DP_LayerCentering())
    layers.append(DP_Flatten())
    layers.append(DP_SpectralDense(units=10, use_bias=False))
    layers.append(DP_ClipGradient(cfg.clip_loss_gradient))

    model = DP_Model(
        layers,
        cfg=cfg,
        noisify_strategy=cfg.noisify_strategy,
        name="mlp_mixer",
    )

    model.build(input_shape=(None, *input_shape))

    return model


def create_VGG(cfg, upper_bound):
    layers = [
        DP_BoundedInput(input_shape=(32, 32, 3), upper_bound=upper_bound),
        DP_SpectralConv2D(
            filters=64,
            kernel_size=5,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        DP_LayerCentering(),
        DP_SpectralConv2D(
            filters=256,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        DP_LayerCentering(),
        DP_SpectralConv2D(
            filters=512,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=4, strides=4),
        DP_Flatten(),
        DP_LayerCentering(),
        DP_SpectralDense(512, use_bias=False),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_SpectralDense(10, use_bias=False),
        DP_AddBias(norm_max=1),
        DP_ClipGradient(cfg.clip_loss_gradient),
    ]

    if cfg.add_biases is False:
        layers = [layer for layer in layers if not isinstance(layer, DP_AddBias)]
    if cfg.layer_centering is False:
        layers = [layer for layer in layers if not isinstance(layer, DP_LayerCentering)]

    model = DP_Sequential(
        layers,
        cfg=cfg,
        noisify_strategy=cfg.noisify_strategy,
    )
    return model
