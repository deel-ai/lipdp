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
from deel.lipdp.layers import DP_LayerCentering
from deel.lipdp.layers import DP_ScaledL2NormPooling2D
from deel.lipdp.layers import DP_SpectralConv2D
from deel.lipdp.layers import DP_SpectralDense
from deel.lipdp.layers import make_residuals
from deel.lipdp.model import DP_Model
from deel.lipdp.model import DP_Sequential


def create_Dense_Model(dp_parameters, dataset_metadata, cfg, upper_bound):
    all_layers = [
        DP_BoundedInput(input_shape=(28, 28, 1), upper_bound=upper_bound),
        DP_Flatten(),
        DP_SpectralDense(1024, use_bias=False, nm_coef=1.0),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_LayerCentering(),
        DP_SpectralDense(512, use_bias=False, nm_coef=1.0),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_LayerCentering(),
        DP_SpectralDense(256, use_bias=False, nm_coef=1.0),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_LayerCentering(),
        DP_SpectralDense(128, use_bias=False, nm_coef=1.0),
        DP_AddBias(norm_max=1),
        DP_GroupSort(2),
        DP_LayerCentering(),
        DP_SpectralDense(10, use_bias=False, nm_coef=1.0),
        DP_AddBias(norm_max=1),
        DP_ClipGradient(epsilon=1, mode="dynamic_svt", patience=10),
    ]
    if not cfg.add_biases:
        model = DP_Sequential(
            [layer for layer in all_layers if not isinstance(layer, DP_AddBias)],
            dp_parameters=dp_parameters,
            dataset_metadata=dataset_metadata,
        )
    elif cfg.add_biases:
        model = DP_Sequential(
            all_layers,
            dp_parameters=dp_parameters,
            dataset_metadata=dataset_metadata,
        )
    return model


def create_ConvNet(dp_parameters, dataset_metadata, cfg, upper_bound):
    all_layers = [
        DP_BoundedInput(input_shape=(28, 28, 1), upper_bound=upper_bound),
        DP_SpectralConv2D(
            filters=16,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=1.0),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        DP_LayerCentering(),
        DP_SpectralConv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=1.0),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        DP_LayerCentering(),
        DP_Flatten(),
        DP_SpectralDense(1024, use_bias=False),
        DP_AddBias(norm_max=1.0),
        DP_SpectralDense(10, use_bias=False),
        DP_AddBias(norm_max=1.0),
        DP_ClipGradient(epsilon=1, mode="dynamic_svt", patience=10),
    ]
    if not cfg.add_biases:
        model = DP_Sequential(
            [layer for layer in all_layers if not isinstance(layer, DP_AddBias)],
            dp_parameters=dp_parameters,
            dataset_metadata=dataset_metadata,
        )
    elif cfg.add_biases:
        model = DP_Sequential(
            all_layers,
            dp_parameters=dp_parameters,
            dataset_metadata=dataset_metadata,
        )
    return model
