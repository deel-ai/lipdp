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
import numpy as np
import tensorflow as tf
from keras.losses import LossFunctionWrapper


@tf.function
def k_cosine_similarity(y_true, y_pred, KX, axis=-1):
    # Cast all values to similar type :
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    KX = tf.cast(KX, dtype=tf.float32)
    # Get minimum value between theoretical and practical value. (TO CHECK)
    factor = tf.where(tf.norm(y_pred) < KX, KX, tf.norm(y_pred))
    y_pred = y_pred / factor
    # Compute and return cosine similarity.
    return -tf.reduce_sum(y_true * y_pred, axis=axis)


class KCosineSimilarity(LossFunctionWrapper):
    def __init__(
        self,
        KX,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="cosine_similarity",
    ):
        super().__init__(
            k_cosine_similarity, KX=KX, reduction=reduction, name=name, axis=axis
        )


# FIRST TRY : TO DEBUG
def get_lip_constant_loss(cfg):
    if cfg.loss in [
        "MulticlassHinge",
        "MulticlassKR",
        "CategoricalCrossentropy",
        "MAE",
    ]:
        L = 1
    elif cfg.loss == "MulticlassHKR":
        L = cfg.alpha + 1
    elif cfg.loss == "TauCategoricalCrossentropy":
        L = cfg.tau * np.sqrt(cfg.num_classes - 1) / cfg.num_classes
    elif cfg.loss == "KCosineSimilarity":
        L = 1 / float(cfg.K * cfg.min_norm)
    else:
        raise TypeError(f"Unrecognised Loss Function Argument {cfg.loss}")
    return L
