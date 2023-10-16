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
import random
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras

from deel.lipdp.layers import DP_ClipGradient


class LossGradientClipping(keras.callbacks.Callback):
    """Updates the clipping value of the last layer of the model."""

    def __init__(self, ds_train, patience, mode, verbose=False):
        super().__init__()
        self.ds_train = ds_train
        self.patience = patience
        self.mode = mode
        self.verbose = verbose

    @abstractmethod
    def _assign_dp_dict(self, last_layer):
        last_layer._dynamic_dp_dict["patience"] = self.patience
        last_layer._dynamic_dp_dict["mode"] = self.mode

    def on_train_begin(self, logs=None):
        last_layer = self.model.layers_backward_order()[0]
        assert isinstance(
            last_layer, DP_ClipGradient
        ), "The last layer of the model must be a DP_ClipGradient layer."

        assert (
            last_layer.mode == "dynamic"
        ), "The mode of the last layer must be dynamic."

        print("On train begin : ")
        initial_value = tf.convert_to_tensor(self.model.loss.get_L(), dtype=tf.float32)
        print(
            "Initial value is now equal to lipschitz constant of loss: ",
            float(initial_value.numpy()),
        )
        last_layer.clip_value.assign(initial_value)
        self._assign_dp_dict(last_layer)

    def get_gradloss(self):
        """Computes the norm of gradient of the loss with respect to the model's output."""
        batch = next(iter(self.ds_train.take(1)))
        imgs, labels = batch
        self.model.loss.reduction = tf.keras.losses.Reduction.NONE
        predictions = self.model(imgs)
        with tf.GradientTape() as tape:
            tape.watch(predictions)
            loss_value = self.model.compiled_loss(labels, predictions)
        self.model.loss.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        grad_loss = tape.gradient(loss_value, predictions)
        norms = tf.norm(grad_loss, axis=-1)
        norms = norms.numpy()
        if self.verbose:
            print("Norms : ", norms)
            print("Max norm: ", np.max(norms))
            print("Quantiles : ", np.quantile(norms, [0.1 * i for i in range(1, 10)]))
        return norms


def clipsum(norms, C):
    """
    Computes the sum of individually clipped elements from the given list or tensor.

    Args:
        norms (list or tensor): A list or tensor containing the elements to be clipped and summed.
        C (float): A clipping constant used to clip the elements.

    Returns:
        float: The sum of the clipped elements.

    Example:
        >>> norms = [1.3, 2.7, 7.5]
        >>> C = 3.0
        >>> clipsum(norms, C)
        7.0
    """
    norms = tf.cast(norms, dtype=tf.float32)
    C = tf.constant([C], dtype=tf.float32)
    return tf.math.reduce_sum(tf.math.minimum(norms, C))


def diff_query(norms, lower, upper, n_points=1000):
    """
    Computes the difference between two sums of clipped elements with two different clipping constants
    on a range of n_points between the lower and upper values.

    Args:
        norms (list or tensor): A list or tensor of values to be clipped and summed.
        lower (float or int): The lower bound of the search range.
        upper (float or int): The upper bound of the search range.
        n_points (int): The number of points between the lower and upper bound.

    Returns:
        alpha (float): The sensitivity of the differentiation query, calculated as (upper - lower) / (n_points - 1).
        points (list): The list of points iterated on between the lower and upper bound.
        queries (float): The values of the difference query over the points range.

    """
    points = np.linspace(lower, upper, num=n_points)
    alpha = (upper - lower) / (n_points - 1)
    queries = []
    for p in points:
        query = clipsum(norms, p) - clipsum(norms, p + alpha)
        queries.append(query)
    return alpha, points, queries


def laplace_above_treshold(queries, sensitivity, T, epsilon):
    """
    SVT inspired algorithm inspired from https://programming-dp.com/ch10.html. Computes
    the index for which the differentiation query of the queries list converges above a
    treshold T. This computation is epsilon-DP.

    Args :
        queries (list or tensor) : list of the values of the difference query.
        sensitivity (float) : sensitivity of the difference computation query.
        T (float) : value of the treshold.
        epsilon (float) : chosen epsilon guarantee on the query.

    Returns :
        ids (int) : the index corresponding the epsilon-DP estimated optimal clipping constant.

    """
    T_hat = T + np.random.laplace(loc=0, scale=2 * sensitivity / epsilon)
    for idx, q in enumerate(queries):
        nu_i = np.random.laplace(loc=0, scale=4 * sensitivity / epsilon)
        if q + nu_i >= T_hat:
            return idx
    return random.randint(0, len(queries) - 1)


class LaplaceAdaptiveLossGradientClipping(LossGradientClipping):
    """Updates the clipping value of the last layer of the model.

    This callback privately updates the clipping value if the last layer
    of the model is a DP_ClipGradient layer with mode = "dynamic".

    Attributes :
        ds_train : a tensorflow dataset object.
    """

    def __init__(self, ds_train, *, patience, epsilon):
        super().__init__(ds_train, patience, "laplace")
        self.epsilon = epsilon

        assert (
            epsilon is not None
        ), "epsilon has to be in constructor for dynamic gradient clipping."
        assert epsilon > 0, "epsilon <= 0 impossible."

    def _assign_dp_dict(self, last_layer):
        super()._assign_dp_dict(last_layer)
        last_layer._dynamic_dp_dict["epsilon"] = self.epsilon

    def on_epoch_end(self, epoch, logs={}):
        # print("Patience : ", epoch % last_layer.patience)
        if epoch % self.patience != 0:
            return
        last_layer = self.model.layers_backward_order()[0]
        norms = self.get_gradloss()
        alpha, points, queries = diff_query(
            norms, lower=0, upper=self.model.loss.get_L()
        )
        T = 0.0  # queries[0] * 0.1 (why?)
        updated_clip_value = points[
            laplace_above_treshold(
                queries, sensitivity=alpha, T=T, epsilon=self.epsilon
            )
        ]
        last_layer.update_clipping_value(updated_clip_value)


class AdaptiveQuantileClipping(LossGradientClipping):
    """Updates the clipping value of the last layer of the model.

    This callback privately updates the clipping value if the last layer
    of the model is a DP_ClipGradient layer with mode = "dynamic".

    This is the canonical implementation proposed in:

        Andrew, G., Thakkar, O., McMahan, B. and Ramaswamy, S., 2021.
        Differentially private learning with adaptive clipping.
        Advances in Neural Information Processing Systems, 34, pp.17455-17466.

    Attributes :
        ds_train : a tensorflow dataset object.
        noise_multiplier : the noise multiplier of private quantile estimation (float).
        quantile : the quantile to estimate (float).
        learning_rate : the learning rate of the exponential gradient step (float).
    """

    def __init__(
        self, ds_train, *, patience, noise_multiplier, quantile, learning_rate
    ):
        super().__init__(ds_train, patience, "quantiles")
        self.noise_multiplier = noise_multiplier
        self.quantile = quantile
        self.learning_rate = learning_rate

    def _assign_dp_dict(self, last_layer):
        super()._assign_dp_dict(last_layer)
        last_layer._dynamic_dp_dict["noise_multiplier"] = self.noise_multiplier

    def on_epoch_end(self, epoch, logs={}):
        # print("Patience : ", epoch % last_layer.patience)
        if epoch % self.patience != 0:
            return
        last_layer = self.model.layers_backward_order()[0]
        norms = self.get_gradloss()
        clip_value = last_layer.clip_value.value()

        # Gaussian mechanism
        avg_above_c_insecure = (norms <= clip_value.numpy()).astype(float)
        sensitivity = 1.0 / len(norms)
        scale_noise = self.noise_multiplier * sensitivity
        noise = np.random.normal(loc=0, scale=scale_noise)
        avg_above_c_private = avg_above_c_insecure.mean() + noise

        # Exponential gradient step
        step = avg_above_c_private - self.quantile
        updated_clip_value = clip_value * np.exp(-self.learning_rate * step)
        last_layer.update_clipping_value(updated_clip_value)
