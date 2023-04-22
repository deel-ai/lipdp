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
    def backpropagate_params(self, input_bound, gradient_bound):
        """Corresponds to the Lipschitz constant of the output wrt the parameters,
            i.e. the norm of the Jacobian of the output wrt the parameters times the norm of the cotangeant vector.

        Args:
            input_bound: Maximum norm of input.
            gradient_bound: Maximum norm of gradients (co-tangent vector)

        Returns:
            Maximum norm of tangent vector."""
        pass

    @abstractmethod
    def backpropagate_inputs(self, input_bound, gradient_bound):
        """Applies the dilatation of the cotangeant vector norm (upstream gradient) by the Jacobian,
            i.e. multiply by the Lipschitz constant of the output wrt input.

        Args:
            input_bound: Maximum norm of input.
            gradient_bound: Maximum norm of gradients (co-tangent vector)

        Returns:
            Maximum norm of tangent vector.
        """
        pass

    @abstractmethod
    def propagate_inputs(self, input_bound):
        """Maximum norm of output.

        Remark: when the layer is linear, this coincides with its Lipschitz constant * input_bound.
        """
        pass

    @abstractmethod
    def has_parameters(self):
        pass


def DP_GNP_Factory(layer_cls):
    """Factory for creating differentially private gradient norm preserving layers that don't have parameters.

    Remark: the layer is assumed to be GNP.
    This means that the gradient norm is preserved by the layer (i.e its Jacobian norm is 1).
    Pllease ensure that the layer is GNP before using this factory.

    Args:
        layer_cls: Class of the layer to wrap.

    Returns:
        A differentially private layer that doesn't have parameters.
    """

    class DP_GNP(layer_cls, DPLayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def backpropagate_params(self, input_bound, gradient_bound):
            raise ValueError("Layer doesn't have parameters")

        def backpropagate_inputs(self, input_bound, gradient_bound):
            return 1 * gradient_bound

        def propagate_inputs(self, input_bound):
            return input_bound

        def has_parameters(self):
            return False

    return DP_GNP


DP_Reshape = DP_GNP_Factory(tf.keras.layers.Reshape)
DP_Lambda = DP_GNP_Factory(tf.keras.layers.Lambda)
DP_Permute = DP_GNP_Factory(tf.keras.layers.Permute)
DP_Flatten = DP_GNP_Factory(tf.keras.layers.Flatten)
DP_GroupSort = DP_GNP_Factory(deel.lip.activations.GroupSort)
DP_InputLayer = DP_GNP_Factory(tf.keras.layers.InputLayer)


class DP_ScaledL2NormPooling2D(deel.lip.layers.ScaledL2NormPooling2D, DPLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("ScaledL2NormPooling2D doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return False


class LayerCentering(tf.keras.layers.Layer):
    def __init__(self, pixelwise=False, channelwise=True, **kwargs):
        self.pixelwise = pixelwise
        self.channelwise = channelwise
        self.axes = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 4:
            if self.channelwise and self.pixelwise:
                raise RuntimeError(
                    "enabling channelwise and pixelwise in LayerCentering "
                    "would average over a single value"
                )
            elif self.channelwise:
                self.axes = [-1]
            elif self.pixelwise:
                self.axes = [1, 2]
            else:
                self.axes = [1, 2, 3]
        else:
            self.axes = range(len(input_shape) - 1)

    @tf.function
    def call(self, inputs, training=True, **kwargs):
        current_means = tf.reduce_mean(inputs, axis=self.axes, keepdims=True)
        return inputs - current_means

    def get_config(self):
        config = {
            "pixelwise": self.pixelwise,
            "channelwise": self.channelwise,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DP_LayerCentering(LayerCentering, DPLayer):
    def __init__(self, *args, **kwargs):
        # TODO - Check that activation has a Jacobian norm of 1
        if "scale" in kwargs and kwargs["scale"]:
            raise ValueError("No scaling allowed.")
        if "center" in kwargs and not kwargs["center"]:
            print("No centering applied, layer is useless.")
        super().__init__(*args, **kwargs)

    def backpropagate_params(self, input_bound, gradient_bound):
        # LAYER IS NOT TRAINABLE RETURNS  input_bound
        raise ValueError("Layer Centering doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return False


class DP_SpectralDense(deel.lip.layers.SpectralDense, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def backpropagate_params(self, input_bound, gradient_bound):
        return input_bound * gradient_bound

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return True


class DP_SpectralConv2D(deel.lip.layers.SpectralConv2D, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def backpropagate_params(self, input_bound, gradient_bound):
        return (
            self._get_coef()
            * np.sqrt(np.prod(self.kernel_size))
            * input_bound
            * gradient_bound
        )

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return True


class DP_SplitResidual(tf.keras.layers.Layer, DPLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwds):
        return (inputs, inputs)

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("Layer doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        assert len(gradient_bound) == 2
        g1, g2 = gradient_bound
        return g1 + g2

    def propagate_inputs(self, input_bound):
        return (input_bound, input_bound)

    def has_parameters(self):
        return False


class DP_MergeResidual(tf.keras.layers.Layer, DPLayer):
    def __init__(self, merge_policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert merge_policy in ["add", "1-lip-add"]
        self.merge_policy = merge_policy

    def call(self, inputs, *args, **kwds):
        assert len(inputs) == 2
        i1, i2 = inputs
        if self.merge_policy == "add":
            return i1 + i2
        elif self.merge_policy == "1-lip-add":
            return 0.5 * (i1 + i2)

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("Layer doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        if self.merge_policy == "add":
            return gradient_bound, gradient_bound
        elif self.merge_policy == "1-lip-add":
            return 0.5 * gradient_bound, 0.5 * gradient_bound

    def propagate_inputs(self, input_bound):
        assert len(input_bound) == 2
        i1, i2 = input_bound
        if self.merge_policy == "add":
            return i1 + i2
        elif self.merge_policy == "1-lip-add":
            return 0.5 * (i1 + i2)

    def has_parameters(self):
        return False


class DP_WrappedResidual(tf.keras.layers.Layer, DPLayer):
    def __init__(self, block, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block

    def call(self, inputs, *args, **kwargs):
        assert len(inputs) == 2
        i1, i2 = inputs
        i2 = self.block(i2, *args, **kwargs)
        return i1, i2

    def backpropagate_params(self, input_bound, gradient_bound):
        assert len(input_bound) == 2
        assert len(gradient_bound) == 2
        _, i2 = input_bound
        _, g2 = gradient_bound
        g2 = self.block.backpropagate_params(i2, g2)
        return g2

    def backpropagate_inputs(self, input_bound, gradient_bound):
        assert len(input_bound) == 2
        assert len(gradient_bound) == 2
        _, i2 = input_bound
        g1, g2 = gradient_bound
        g2 = self.block.backpropagate_inputs(i2, g2)
        return g1, g2

    def propagate_inputs(self, input_bound):
        assert len(input_bound) == 2
        i1, i2 = input_bound
        i2 = self.block.propagate_inputs(i2)
        return i1, i2

    def has_parameters(self):
        return self.block.has_parameters()

    @property
    def nm_coef(self):
        """Returns the norm multiplier coefficient of the layer.

        Remark: this is a property to mimic the behavior of an attribute.
        """
        return self.block.nm_coef


def make_residuals(merge_policy, wrapped_layers):
    layers = [DP_SplitResidual()]

    for layer in wrapped_layers:
        residual_block = DP_WrappedResidual(layer)
        layers.append(residual_block)

    layers.append(DP_MergeResidual(merge_policy))

    return layers
