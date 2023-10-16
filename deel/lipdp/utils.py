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


class ScaledAUC(tf.keras.metrics.AUC):
    def __init__(self, scale, name="auc", **kwargs):
        if "from_logits" in kwargs and kwargs["from_logits"] is False:
            raise ValueError("ScaledAUC must be used with from_logits=True")
        kwargs["from_logits"] = True
        super().__init__(name=name, **kwargs)
        self.scale = scale

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred * self.scale
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


class CertifiableAUROC(tf.keras.metrics.AUC):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred is 1-Lipschitz wrt the input and labels are in {-1, 1}
        labels = 2 * tf.cast(y_true, tf.float32) - 1
        y_pred = y_pred - labels * self.radius
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


class PrivacyMetrics(tf.keras.callbacks.Callback):
    """Callback to compute privacy metrics at the end training.

    Modified from official tutorial https://www.tensorflow.org/responsible_ai/privacy/tutorials/privacy_report

    Args:
        np_dataset: The dataset used to train the model. It must be a tuple (x_train, y_train, x_test, y_test).
    """

    def __init__(self, np_dataset, log_fn="all"):
        super().__init__()
        if log_fn == "wandb":
            import wandb

            log_fn = wandb.log
        elif log_fn == "logging":
            import logging

            log_fn = logging.info
        elif log_fn == "all":
            import wandb
            import logging

            log_fn = lambda x: [wandb.log(x), logging.info(x)]
        else:
            raise ValueError(f"Unknown log_fn {log_fn}")
        self.log_fn = log_fn

        x_train, y_train, x_test, y_test = np_dataset
        self.x_train = x_train
        self.x_test = x_test
        self.labels_train = y_train
        self.labels_test = y_test
        try:
            import tensorflow_privacy
            from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import (
                membership_inference_attack as mia,
            )
            import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures as mia_ds

            self.mia = mia
            self.mia_ds = mia_ds
        except ImportError:
            self.mia = None
            raise ImportError(
                "tensorflow_privacy is not installed. Please install it to use PrivacyMetrics."
            )
        self.attack_results = None

    def on_train_end(self, logs=None):
        print(f"\nRunning privacy report...")

        logits_train = self.model.predict(self.x_train, batch_size=2000)
        logits_test = self.model.predict(self.x_test, batch_size=2000)

        print(f"prob_train.shape = {logits_train.shape}")
        print(f"prob_test.shape = {logits_test.shape}")
        print(f"label_train.shape = {self.labels_train.shape}")
        print(f"label_test.shape = {self.labels_test.shape}")

        attack_results = self.mia.run_attacks(
            self.mia_ds.AttackInputData(
                labels_train=self.labels_train,
                labels_test=self.labels_test,
                logits_train=logits_train,
                logits_test=logits_test,
            ),
            self.mia_ds.SlicingSpec(entire_dataset=True, by_class=True),
            attack_types=(
                self.mia_ds.AttackType.THRESHOLD_ATTACK,
                self.mia_ds.AttackType.LOGISTIC_REGRESSION,
            ),
        )

        self.attack_results = attack_results

    def log_report(self):
        """Prints the privacy report."""
        attack_results = self.attack_results
        summary = attack_results.calculate_pd_dataframe()
        print(summary)
        entire_dataset = summary[summary["slice feature"] == "Entire dataset"]
        per_class = summary[summary["slice feature"] == "class"]
        max_auc_entire_dataset = entire_dataset["AUC"].max()
        max_adv_entire_dataset = entire_dataset["Attacker advantage"].max()
        max_auc_per_class = per_class["AUC"].max()
        max_adv_per_class = per_class["Attacker advantage"].max()
        to_log = {
            "mia_auc_per_class": max_auc_per_class,
            "mia_adv_per_class": max_adv_per_class,
            "mia_auc_entire_dataset": max_auc_entire_dataset,
            "mia_adv_entire_dataset": max_adv_entire_dataset,
        }
        self.log_fn(to_log)
