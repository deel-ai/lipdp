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
import math

import tensorflow as tf
from deel.lip import losses
from tensorflow.python.keras.losses import Loss


class DP_Loss(Loss):
    def get_L(self):
        """returns the lipschitz constant of the loss"""
        raise NotImplementedError()


class DP_KCosineSimilarity(DP_Loss):
    def __init__(
        self,
        K=1.0,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="cosine_similarity",
    ):
        super().__init__(reduction=reduction, name=name)
        # as the espilon is applied before the sqrt in tf.linalg.l2_normalize we
        # apply square to it
        self.K = K**2
        self.axis = axis

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.linalg.l2_normalize(y_true, epsilon=self.K, axis=self.axis)
        y_pred = tf.linalg.l2_normalize(y_pred, epsilon=self.K, axis=self.axis)
        return -tf.reduce_sum(y_true * y_pred, axis=self.axis)

    def get_L(self):
        """returns the lipschitz constant of the loss"""
        return 1 / float(self.K)


class DP_TauCategoricalCrossentropy(losses.TauCategoricalCrossentropy, DP_Loss):
    def __init__(
            self,
            tau,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name="TauCategoricalCrossentropy",
    ):
        """
        Similar to original categorical crossentropy, but with a settable temperature
        parameter.

        Args:
            tau (float): temperature parameter.
            reduction: reduction of the loss, must be SUM_OVER_BATCH_SIZE in order have a correct accounting.
            name (str): name of the loss
        """
        super(DP_TauCategoricalCrossentropy, self).__init__(
            tau=tau, reduction=reduction, name=name
        )

    def get_L(self):
        """returns the lipschitz constant of the loss"""
        # as the implementation divide the loss by self.tau (and as it is used with "from_logit=True")
        return math.sqrt(2)


class DP_MulticlassHKR(losses.MulticlassHKR, DP_Loss):
    def __init__(
            self,
            alpha=10.0,
            min_margin=1.0,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name="MulticlassHKR",
    ):
        """
        The multiclass version of HKR. This is done by computing the HKR term over each
        class and averaging the results.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            alpha (float): regularization factor
            min_margin (float): margin to enforce.
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name (str): passed to tf.keras.Loss constructor

        """
        super(DP_MulticlassHKR, self).__init__(
            alpha=alpha,
            min_margin=min_margin,
            multi_gpu=False,
            reduction=reduction,
            name=name,
        )

    def get_L(self):
        """returns the lipschitz constant of the loss"""
        return self.alpha + 1.0


class DP_MulticlassHinge(losses.MulticlassHinge, DP_Loss):
    def __init__(
            self,
            min_margin=1.0,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name="MulticlassHinge",
    ):
        """
        Loss to estimate the Hinge loss in a multiclass setup. It computes the
        element-wise Hinge term. Note that this formulation differs from the one
        commonly found in tensorflow/pytorch (which maximises the difference between
        the two largest logits). This formulation is consistent with the binary
        classification loss used in a multiclass fashion.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            min_margin (float): margin to enforce.
            reduction: passed to tf.keras.Loss constructor
            name (str): passed to tf.keras.Loss constructor

        """
        super(DP_MulticlassHinge, self).__init__(
            min_margin=min_margin, reduction=reduction, name=name
        )

    def get_L(self):
        """returns the lipschitz constant of the loss"""
        return 1.0


class DP_MulticlassKR(losses.MulticlassKR, DP_Loss):
    def __init__(
            self,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name="MulticlassKR",
    ):
        r"""
        Loss to estimate average of Wasserstein-1 distance using Kantorovich-Rubinstein
        duality over outputs. In this multiclass setup, the KR term is computed for each
        class and then averaged.

        Note that `y_true` should be one-hot encoded or pre-processed with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Using a multi-GPU/TPU strategy requires to set `multi_gpu` to True and to
        pre-process the labels `y_true` with the
        `deel.lip.utils.process_labels_for_multi_gpu()` function.

        Args:
            multi_gpu (bool): set to True when running on multi-GPU/TPU
            reduction: passed to tf.keras.Loss constructor
            name (str): passed to tf.keras.Loss constructor

        """
        super(DP_MulticlassKR, self).__init__(reduction=reduction, name=name)

    def get_L(self):
        """returns the lipschitz constant of the loss"""
        return 1.0


class DP_MeanAbsoluteError(tf.keras.losses.MeanAbsoluteError, DP_Loss):
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="MulticlassKR",
    ):
        r"""
        Mean Absolute Error
        """
        super(DP_MeanAbsoluteError, self).__init__(reduction=reduction, name=name)

    def get_L(self):
        """returns the lipschitz constant of the loss"""
        return 1.0
