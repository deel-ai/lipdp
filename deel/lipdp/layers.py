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

import deel.lip
from deel.lip.constraints import SpectralConstraint
from deel.lip.normalizers import DEFAULT_EPS_BJORCK


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
        """Maximum norm of output of element.

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
    Please ensure that the layer is GNP before using this factory.

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

    DP_GNP.__name__ = f"DP_{layer_cls.__name__}"
    return DP_GNP


DP_Reshape = DP_GNP_Factory(tf.keras.layers.Reshape)
DP_Lambda = DP_GNP_Factory(tf.keras.layers.Lambda)
DP_Permute = DP_GNP_Factory(tf.keras.layers.Permute)
DP_Flatten = DP_GNP_Factory(tf.keras.layers.Flatten)
DP_GroupSort = DP_GNP_Factory(deel.lip.activations.GroupSort)
DP_ReLU = DP_GNP_Factory(tf.keras.layers.ReLU)
DP_InputLayer = DP_GNP_Factory(tf.keras.layers.InputLayer)
DP_ScaledGlobalL2NormPooling2D = DP_GNP_Factory(
    deel.lip.layers.ScaledGlobalL2NormPooling2D
)


class DP_MaxPool2D(tf.keras.layers.MaxPool2D, DPLayer):
    """Max pooling layer that preserves the gradient norm.

    Args:
        layer_cls: Class of the layer to wrap.

    Returns:
        A differentially private layer that doesn't have parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.strides is None or self.strides == self.pool_size
        ), "Ensure that strides == pool_size, otherwise it is not 1-Lipschitz."

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("Layer doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return False


class DP_ScaledL2NormPooling2D(deel.lip.layers.ScaledL2NormPooling2D, DPLayer):
    """Max pooling layer that preserves the gradient norm.

    Args:
        layer_cls: Class of the layer to wrap.

    Returns:
        A differentially private layer that doesn't have parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.strides is None or self.strides == self.pool_size
        ), "Ensure that strides == pool_size, otherwise it is not 1-Lipschitz."

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("Layer doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return False


class DP_BoundedInput(tf.keras.layers.Layer, DPLayer):
    """Input layer that clips the input to a given norm.

    Remark: every pipeline should start with this layer.

    Attributes:
        upper_bound: Maximum norm of the input.
        enforce_clipping: If True (default), the input is clipped to the given norm.
    """

    def __init__(self, *args, upper_bound, enforce_clipping=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.upper_bound = tf.convert_to_tensor(upper_bound)
        self.enforce_clipping = enforce_clipping

    def call(self, x, *args, **kwargs):
        if self.enforce_clipping:
            axes = list(range(1, len(x.shape)))
            x = tf.clip_by_norm(x, self.upper_bound, axes=axes)
        return x

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("InputLayer doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        if input_bound is None:
            return self.upper_bound
        return tf.math.minimum(self.upper_bound, input_bound)

    def has_parameters(self):
        return False


class DP_ScaledL2NormPooling1D(tf.keras.layers.Layer, DPLayer):
    def __init__(self, axis=-1, eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = tf.convert_to_tensor(eps)
        self.axis = axis

    def call(self, inputs, training=True, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=self.axis) + self.eps)

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("DP_ScaledL2NormPooling1D doesn't have parameters")

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


class DP_QuickSpectralDense(tf.keras.layers.Dense, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        kwargs.update(
            dict(
                kernel_initializer="orthogonal",
                kernel_constraint="deel-lip>SpectralConstraint",
            )
        )
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


class DP_QuickFrobeniusDense(tf.keras.layers.Dense, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        kwargs.update(
            dict(
                kernel_initializer="orthogonal",
                kernel_constraint="deel-lip>FrobeniusConstraint",
            )
        )
        # Remark: the Frobenius constraint is applied on the whole matrix,
        # not on each row. Therefore the Lipschitz constant is 1 since:
        # ||A||_2 <= ||A||_F = 1
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


def _compute_conv_lip_factor(kernel_size, strides, input_shape, data_format):
    """Compute the Lipschitz factor to apply on estimated Lipschitz constant in
    convolutional layer. This factor depends on the kernel size, the strides and the
    input shape.

    Copied from deel-lip.
    """
    stride = np.prod(strides)
    kh, kw = kernel_size[0], kernel_size[1]
    kh_div2 = (kh - 1) / 2
    kw_div2 = (kw - 1) / 2

    if data_format == "channels_last":
        h, w = input_shape[-3], input_shape[-2]
    elif data_format == "channels_first":
        h, w = input_shape[-2], input_shape[-1]
    else:
        raise RuntimeError("data_format not understood: " % data_format)

    if stride == 1:
        return np.sqrt(
            (w * h)
            / ((kh * h - kh_div2 * (kh_div2 + 1)) * (kw * w - kw_div2 * (kw_div2 + 1)))
        )
    else:
        return np.sqrt(1.0 / (np.ceil(kh / strides[0]) * np.ceil(kw / strides[1])))


class DP_QuickSpectralConv2D(tf.keras.layers.Conv2D, DPLayer):
    def __init__(self, *args, nm_coef=1, orthogonal=True, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        kwargs["use_bias"] = False
        eps_bjorck = DEFAULT_EPS_BJORCK if orthogonal else None
        constraint = SpectralConstraint(eps_bjorck=eps_bjorck)
        kwargs.update(
            dict(
                kernel_initializer="orthogonal",
                kernel_constraint=constraint,
            )
        )
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def _get_coef(self):
        return _compute_conv_lip_factor(
            self.kernel_size, self.strides, self.input_shape, self.data_format
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True
        self.kernel.constraint.k_coef_lip = _compute_conv_lip_factor(
            self.kernel_size, self.strides, input_shape, self.data_format
        )
        self.kernel.assign(self.kernel.constraint(self.kernel))

    def call(self, inputs):
        return super().call(inputs)

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


class DP_SpectralConv2D(deel.lip.layers.SpectralConv2D, DPLayer):
    def __init__(self, *args, nm_coef=1, **kwargs):
        if "use_bias" in kwargs and kwargs["use_bias"]:
            raise ValueError("No bias allowed.")
        if "trainable" in kwargs and not kwargs["trainable"]:
            self.trainable = kwargs["trainable"]
        else:
            self.trainable = True
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def backpropagate_params(self, input_bound, gradient_bound):
        return (
            tf.convert_to_tensor(self._get_coef(), dtype=tf.float32)
            * tf.convert_to_tensor(np.sqrt(np.prod(self.kernel_size)), dtype=tf.float32)
            * input_bound
            * gradient_bound
        )

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return self.trainable


@tf.custom_gradient
def clip_gradient(x, clip_value):
    """Clips the gradient during the backward pass.

    Behave like identity function during the forward pass.
    """

    def grad_fn(dy):
        # clip by norm each row
        axes = list(range(1, len(dy.shape)))
        clipped_dy = tf.clip_by_norm(dy, clip_value, axes=axes)
        return clipped_dy, None  # No gradient for clip_value

    return x, grad_fn


class DP_ClipGradient(tf.keras.layers.Layer, DPLayer):
    """Clips the gradient during the backward pass.

    Behaves like identity function during the forward pass.
    The clipping is done automatically during the backward pass.

    Attributes:
        clip_value (float, optional):
                            The maximum norm of the gradient allowed. Only
                            declare this variable if you plan on using the "fixed" clipping mode.
                            Otherwise it will be updated automatically.
        mode (str): The mode of clipping. Either "fixed" or "dynamic". Default is "fixed".

    Warning : The mode "dynamic" needs to be used along a callback that updates the clipping value.
    """

    def __init__(self, clip_value, mode="fixed", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dynamic_dp_dict = {}  # to be filled by the callback

        assert mode in ["fixed", "dynamic"]
        self.mode = mode

        assert clip_value is None or clip_value >= 0, "clip_value must be positive"
        if mode == "fixed":
            assert (
                clip_value is not None
            ), "clip_value must be declared when using the fixed mode"

        if clip_value is None:
            clip_value = (
                0.0  # Change type back to float in case clip_value needs to be updated
            )

        self.clip_value = tf.Variable(clip_value, trainable=False, dtype=tf.float32)

    def update_clipping_value(self, new_clip_value):
        print("Update clipping value to : ", float(new_clip_value.numpy()))
        self.clip_value.assign(new_clip_value)

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.convert_to_tensor(tf.cast(tf.shape(inputs)[0], tf.float32))
        # the clipping is done elementwise
        # since REDUCTION=SUM_OVER_BATCH_SIZE, we need to divide by batch_size
        # to get the correct norm.
        # this makes the clipping independent of the batch size.
        elementwise_clip_value = self.clip_value.value() / batch_size
        return clip_gradient(inputs, elementwise_clip_value)

    def backpropagate_params(self, input_bound, gradient_bound):
        raise ValueError("ClipGradient doesn't have parameters")

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return tf.math.minimum(gradient_bound, self.clip_value)

    def propagate_inputs(self, input_bound):
        return input_bound

    def has_parameters(self):
        return False


class AddBias(tf.keras.layers.Layer):
    """Adds a bias to the input.

    Remark: the euclidean norm of the bias must be bounded in advance.
    Note that this is the euclidean norm of the whole bias vector, not
    the norm of each element of the bias vector.

    Warning: beware zero gradients outside the ball of norm norm_max.
    In the future, we might choose a smoother projection on the ball to ensure
    that the gradient remains non zero outside the ball.
    """

    def __init__(self, norm_max, **kwargs):
        super().__init__(**kwargs)
        self.norm_max = tf.convert_to_tensor(norm_max)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        # parametrize the bias so it belongs to a ball of norm norm_max.
        bias = tf.convert_to_tensor(
            tf.clip_by_norm(self.bias, self.norm_max)
        )  # 1-Lipschitz operation.
        return inputs + bias


class DP_AddBias(AddBias, DPLayer):
    """Adds a bias to the input.

    The bias is projected on the ball of norm `norm_max` during training.
    The projection on the ball is a 1-Lipschitz function, since the ball
    is convex.
    """

    def __init__(self, *args, nm_coef=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.nm_coef = nm_coef

    def backpropagate_params(self, input_bound, gradient_bound):
        return gradient_bound  # clipping is a 1-Lipschitz operation.

    def backpropagate_inputs(self, input_bound, gradient_bound):
        return 1 * gradient_bound  # adding is a 1-Lipschitz operation.

    def propagate_inputs(self, input_bound):
        return input_bound + self.norm_max

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
    """Returns a list of layers that implement a residual block.

    Args:
        merge_policy: either "add" or "1-lip-add".
        wrapped_layers: a list of layers that will be wrapped in residual blocks.

    Returns:
        A list of layers that implement a residual block.
    """
    layers = [DP_SplitResidual()]

    for layer in wrapped_layers:
        residual_block = DP_WrappedResidual(layer)
        layers.append(residual_block)

    layers.append(DP_MergeResidual(merge_policy))

    return layers
