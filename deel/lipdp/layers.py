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
from abc import abstractmethod

import numpy as np
import tensorflow as tf

import deel


class DPLayer:
    """
    Wrapper for created differentially private layers, instanciates abstract methods
    use for computing the bounds of the gradient relatively to the parameters and to the
    input.
    """

    @abstractmethod
    def get_DP_LipCoef_params(self, input_bounds):
        pass

    @abstractmethod
    def get_DP_LipCoef_inputs(self, input_bounds):
        pass

    @abstractmethod
    def has_parameters(self):
        pass


class DP_LayerCentering(tf.keras.layers.LayerNormalization, DPLayer):
    def __init__(self, *args, **kwargs):
        # TODO - Check that activation has a Jacobian norm of 1
        if "scale" in kwargs and kwargs["scale"]:
            raise ValueError("No scaling allowed.")
        if "center" in kwargs and not kwargs["center"]:
            print("No centering applied, layer is useless.")
        super().__init__(*args, **kwargs)

    def get_DP_LipCoef_params(self, input_bounds):
        # LAYER IS NOT TRAINABLE RETURNS 1 * INPUT_BOUNDS
        raise ValueError("Layer Centering doesn't have parameters")

    def get_DP_LipCoef_inputs(self, input_bounds):
        return 1 * input_bounds

    def has_parameters(self):
        return False


class DP_ResidualSpectralDense(deel.lip.layers.SpectralDense, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        # TODO - Check that activation has a Jacobian norm of 1
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def get_DP_LipCoef_params(self, input_bounds):
        return 0.5 * input_bounds

    def get_DP_LipCoef_inputs(self, input_bounds):
        return 1 * input_bounds

    def has_parameters(self):
        return True


class DP_SpectralDense(deel.lip.layers.SpectralDense, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        # TODO - Check that activation has a Jacobian norm of 1
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def get_DP_LipCoef_params(self, input_bounds):
        return 1 * input_bounds

    def get_DP_LipCoef_inputs(self, input_bounds):
        return 1 * input_bounds

    def has_parameters(self):
        return True


class DP_SpectralConv2D(deel.lip.layers.SpectralConv2D, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def get_DP_LipCoef_params(self, input_bounds):
        return self._get_coef() * np.sqrt(np.prod(self.kernel_size)) * input_bounds

    def get_DP_LipCoef_inputs(self, input_bounds):
        return 1 * input_bounds

    def has_parameters(self):
        return True
