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
from absl.testing import absltest
from absl.testing import parameterized

from deel.lipdp import losses


class LossesTest(parameterized.TestCase):
    def _get_preds(self, bs: int, n_classes: int, seed: int):
        y_true = np.eye(n_classes)
        y_true = np.concatenate([y_true] * bs, axis=0)
        np.random.seed(seed)
        y_pred = np.random.uniform(size=(len(y_true), n_classes))
        return tf.constant(y_true), tf.constant(y_pred, dtype=tf.float32)

    def _test_grad_bounds(self, loss, y_true, y_pred):
        with tf.GradientTape() as tape:
            tape.watch(y_pred)
            loss_value = loss(y_true, y_pred)
        grad = tape.gradient(loss_value, y_pred)
        grad_norms = tf.norm(grad, axis=-1)

        atol = 1e-7
        for grad_norm in grad_norms:
            self.assertLessEqual(grad_norm, loss.get_L() + atol)

    @parameterized.parameters(
        (0.1,),
        (1.0,),
        (10.0,),
    )
    def test_tau_cce(self, tau: float):
        loss = losses.DP_TauCategoricalCrossentropy(tau=tau)
        y_true, y_pred = self._get_preds(bs=16, n_classes=10, seed=1337)
        self._test_grad_bounds(loss, y_true, y_pred)

    @parameterized.parameters(
        (0.1,),
        (1.0,),
        (10.0,),
    )
    def test_k_cosine_similarity(self, K: float):
        loss = losses.DP_KCosineSimilarity(K=K)
        y_true, y_pred = self._get_preds(bs=16, n_classes=10, seed=896)
        y_true = tf.cast(y_true, dtype=tf.float32)
        self._test_grad_bounds(loss, y_true, y_pred)

    @parameterized.parameters(
        (0.1,),
        (1.0,),
        (10.0,),
    )
    def test_multiclass_hkr(self, alpha: float):
        loss = losses.DP_MulticlassHKR(alpha=alpha)
        y_true, y_pred = self._get_preds(bs=16, n_classes=10, seed=123)
        self._test_grad_bounds(loss, y_true, y_pred)

    @parameterized.parameters(
        (0.1,),
        (1.0,),
        (10.0,),
    )
    def test_multiclass_hinge(self, margin: float):
        loss = losses.DP_MulticlassHinge(min_margin=margin)
        y_true, y_pred = self._get_preds(bs=16, n_classes=10, seed=123)
        self._test_grad_bounds(loss, y_true, y_pred)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    absltest.main()
