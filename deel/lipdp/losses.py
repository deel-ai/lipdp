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
from tensorflow.python.keras.losses import Loss


class KCosineSimilarity(Loss):
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
        self.K = K ** 2
        self.axis = axis

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.linalg.l2_normalize(y_true, epsilon=self.K, axis=self.axis)
        y_pred = tf.linalg.l2_normalize(y_pred, epsilon=self.K, axis=self.axis)
        return -tf.reduce_sum(y_true * y_pred, axis=self.axis)


# FIRST TRY : TO DEBUG
def get_lip_constant_loss(cfg, input_bound):
    """Get the maximum norm of elementwise gradients, taking into account the renormalization factor consequent to the batch size.

    Args:
        cfg (dict): Configuration dictionary.
        input_bound (float): Bound on the input of the loss (i.e norm of the output of the last layer).

    Returns:
        float: Lipschitz constant of the loss function.
    """
    if cfg.loss in [
        "MulticlassHinge",
        "MulticlassKR",
        "MAE",
    ]:
        L = 1
    elif cfg.loss == "MulticlassHKR":
        L = cfg.alpha + 1
    elif cfg.loss == "TauCategoricalCrossentropy":
        L = math.sqrt(2)
    elif cfg.loss == "KCosineSimilarity":
        L = 1 / float(cfg.K)
    else:
        raise TypeError(f"Unrecognised Loss Function Argument {cfg.loss}")
    return L / cfg.batch_size
