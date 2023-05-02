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
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
    compute_noise,
)


def get_sweep_config(cfg):
    # Define pertinent parameters according to config :

    header_to_all_sweeps = {
        "method": "bayes",
        "name": "default",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
    }

    common_hyper_parameters = {
        "input_clipping": {
            "max": 1.0,
            "min": 0.1,
            "distribution": "log_uniform_values",
        },
        "clip_loss_gradient": {
            "max": 10.0,
            "min": 1e-6,
            "distribution": "log_uniform_values",
        },
        "batch_size": {"values": [5_000, 10_000], "distribution": "categorical"},
        "steps": {"values": [150, 300, 500], "distribution": "categorical"},
        "add_biases": {"values": [True, False], "distribution": "categorical"},
    }

    learning_rate_SGD = {
        "learning_rate": {
            "max": 0.1,
            "min": 0.001,
            "distribution": "log_uniform_values",
        },
    }

    learning_rate_Adam = {
        "learning_rate": {
            "max": 0.01,
            "min": 0.0001,
            "distribution": "log_uniform_values",
        },
    }

    if cfg.loss == "MulticlassHinge":
        parameters_loss = {
            "min_margin": {
                "max": 3.0,
                "min": 0.001,
                "distribution": "log_uniform_values",
            },
        }

    elif cfg.loss == "MulticlassHKR":
        parameters_loss = {
            "alpha": {"max": 2000.0, "min": 0.01, "distribution": "log_uniform_values"},
            "min_margin": {
                "max": 1.0,
                "min": 0.001,
                "distribution": "log_uniform_values",
            },
        }

    elif cfg.loss == "MulticlassKR":
        parameters_loss = {}

    elif cfg.loss == "MAE":
        parameters_loss = {}

    elif cfg.loss == "TauCategoricalCrossentropy":
        parameters_loss = {
            "tau": {"max": 18.0, "min": 0.001, "distribution": "log_uniform_values"},
        }

    elif cfg.loss == "KCosineSimilarity":
        parameters_loss = {
            "K": {"max": 1.0, "min": 0.01, "distribution": "log_uniform_values"},
        }

    else:
        print("Unrecognised loss functions")

    learning_rate_parameters = (
        learning_rate_SGD if cfg.optimizer == "SGD" else learning_rate_Adam
    )

    assert common_hyper_parameters.keys().isdisjoint(parameters_loss)
    assert common_hyper_parameters.keys().isdisjoint(learning_rate_parameters)
    assert parameters_loss.keys().isdisjoint(learning_rate_parameters)

    sweep_config = {
        **header_to_all_sweeps,
        "parameters": {
            **common_hyper_parameters,
            **parameters_loss,
            **learning_rate_parameters,
        },
    }

    epochs = cfg.steps // (cfg.N // cfg.batch_size)

    # Handle sweep
    sweep_name = cfg.log_wandb[len("sweep_") :]
    sweep_config["name"] = sweep_name
    for key, value in cfg.items():
        if key not in sweep_config["parameters"]:
            if key == "loss":
                print("Loss : ", value)
            sweep_config["parameters"][key] = {
                "value": value,
                "distribution": "constant",
            }
    # Return the config of sweep :
    return sweep_config
