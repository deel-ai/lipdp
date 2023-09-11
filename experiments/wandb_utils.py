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
import os
from typing import Callable

import wandb
import yaml
from ml_collections.config_dict import ConfigDict


def init_wandb(cfg: ConfigDict, project: str):
    if cfg.log_wandb == "run":
        wandb.init(project=project, mode="online", config=cfg)

    elif cfg.log_wandb == "disabled":
        wandb.init(project=project, mode="disabled", config=cfg)

    elif cfg.log_wandb.startswith("sweep_"):
        wandb.init()
        for key, value in wandb.config.items():
            cfg[key] = value


def run_with_wandb(cfg: ConfigDict, train_function: Callable, project: str):
    """Run an individual run or a sweep."""
    wandb.login()
    # indivudal run
    if cfg.log_wandb in ["run", "disabled"]:
        train_function()
    # run sweep
    elif cfg.log_wandb.startswith("sweep_"):
        if cfg.sweep_id == "":
            if cfg.sweep_yaml_config != "":
                # load yaml config
                assert os.path.exists(cfg.sweep_yaml_config)
                sweep_config = _get_sweep_config_from_yaml(cfg)
            else:
                # default sweep config
                sweep_config = _get_default_sweep_config(cfg)
            sweep_id = wandb.sweep(sweep=sweep_config, project=project)
        else:
            sweep_id = cfg.sweep_id
        wandb.agent(
            sweep_id, function=train_function, project=project, count=cfg.opt_iterations
        )


def _sanitize_sweep_config_from_cfg(sweep_config: dict, cfg: ConfigDict) -> dict:
    """
    Name the sweep config and add default values for unspecified parameters.
    """
    # sweep name
    sweep_name = cfg.log_wandb[len("sweep_") :]
    sweep_config["name"] = sweep_name
    # get unspecified params from cfg
    for key, value in cfg.items():
        if key not in sweep_config["parameters"]:
            if key == "loss":
                print("Loss : ", value)
            sweep_config["parameters"][key] = {
                "value": value,
                "distribution": "constant",
            }
    return sweep_config


def _get_sweep_config_from_yaml(cfg):
    """
    Load sweep config from yaml file.
    """
    # load sweep config from yaml
    with open(cfg.sweep_yaml_config, "r") as f:
        sweep_config = yaml.safe_load(f)

    # complete sweep config with unspecified cfg constant params
    sweep_config = _sanitize_sweep_config_from_cfg(sweep_config, cfg)
    return sweep_config


def _get_default_sweep_config(cfg):
    """
    Get default sweep config.
    """
    # Define pertinent parameters according to config :

    header_to_all_sweeps = {
        "method": "bayes",
        "name": "default",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        # "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        # maybe a bit dangerous to use with automatic tuning of the number of epochs?
        # ideally we would want the early stopping to happen on "epsilon" and not on the number of steps.
        # otherwise the runs with hundred of epochs will be early stopped despite having small values of epsilon.
    }

    common_hyper_parameters = {
        "input_bound": {
            "max": 200.0,
            "min": 0.01,
            "distribution": "log_uniform_values",
        },
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

    if cfg.loss == "TauCategoricalCrossentropy":
        parameters_loss = {
            "tau": {"max": 200.0, "min": 0.001, "distribution": "log_uniform_values"},
        }

    elif cfg.loss == "KCosineSimilarity":
        parameters_loss = {
            "K": {"max": 200.0, "min": 0.001, "distribution": "log_uniform_values"},
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

    # complete sweep config with unspecified cfg constant params
    sweep_config = _sanitize_sweep_config_from_cfg(sweep_config, cfg)
    return sweep_config
