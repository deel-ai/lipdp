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

import numpy as np
import tensorflow as tf

from autodp import mechanism_zoo
from autodp import transformer_zoo
from autodp.autodp_core import Mechanism
from tensorflow import keras

import deel
import wandb
from deel.lipdp.layers import DPLayer
from deel.lipdp.losses import get_lip_constant_loss


class DP_Accountant(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        niter = (epoch + 1) * math.ceil(self.model.cfg.N / self.model.cfg.batch_size)
        epsilon, delta = get_eps_delta(model=self.model, niter=niter)
        print(f"\n {(epsilon,delta)}-DP guarantees for epoch {epoch+1} \n")
        wandb.log({"epsilon": epsilon})

class Global_DPGD_Mechanism(Mechanism):
    def __init__(self, prob, sigma, niter, delta, name="Layer_DPGD"):
        # Init
        Mechanism.__init__(self)
        self.name = name
        self.params = {"prob": prob, "sigmas": sigma, "niter": niter}

        model_mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
        subsample = transformer_zoo.AmplificationBySampling()
        SubsampledModelGaussian_mech = subsample(
            model_mech, prob, improved_bound_flag=True
        )
        compose = transformer_zoo.Composition()
        global_mech = compose([SubsampledModelGaussian_mech], [niter])

        # Get relevant information
        self.epsilon = global_mech.get_approxDP(delta=delta)
        self.delta = delta

        # Propagate updates
        rdp_global = global_mech.RenyiDP
        self.propagate_updates(rdp_global, type_of_update="RDP")


class Local_DPGD_Mechanism(Mechanism):
    def __init__(self, prob, sigmas, niter, delta, name="Layer_DPGD"):
        # Init
        Mechanism.__init__(self)
        self.name = name
        self.params = {"prob": prob, "sigmas": sigmas, "niter": niter}

        # Create useful vars
        layer_mechanisms = []

        for sigma in sigmas:
            mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
            layer_mechanisms.append(mech)

        # Accountant composition on layers
        compose_gaussians = transformer_zoo.ComposeGaussian()
        composed_layers = compose_gaussians(layer_mechanisms, [1] * len(sigmas))

        # Accountant composition of subsampling processes
        subsample = transformer_zoo.AmplificationBySampling()
        SubsampledGaussian_mech = subsample(
            composed_layers, prob, improved_bound_flag=True
        )

        # Accounting for niter iterations
        compose = transformer_zoo.Composition()
        global_mech = compose([SubsampledGaussian_mech], [niter])

        # Get relevant information
        self.epsilon = global_mech.get_approxDP(delta=delta)
        self.delta = delta

        # Propagate updates
        rdp_global = global_mech.RenyiDP
        self.propagate_updates(rdp_global, type_of_update="RDP")


def get_nm_coefs(model):
    dict_coefs = {}
    for layer in model.layers_forward_order():
        assert isinstance(layer, DPLayer)
        if layer.has_parameters():
            assert len(layer.trainable_variables) == 1
            dict_coefs[layer.trainable_variables[0].name] = layer.nm_coef
    return dict_coefs


def get_eps_delta(model, niter):
    prob = model.cfg.batch_size / model.cfg.N
    # nm_coefs.values is seamingly in the right order :
    if model.cfg.noisify_strategy == "local":
        nm_coefs = get_nm_coefs(model)
        sigmas = [model.cfg.noise_multiplier * coef for coef in nm_coefs.values()]
        mech = Local_DPGD_Mechanism(
            prob=prob, sigmas=sigmas, delta=model.cfg.delta, niter=niter
        )
    if model.cfg.noisify_strategy == "global":
        mech = Global_DPGD_Mechanism(
            prob=prob,
            sigma=model.cfg.noise_multiplier,
            delta=model.cfg.delta,
            niter=niter,
        )
    return mech.epsilon, mech.delta


def compute_gradient_bounds(model):
    # Initialisation, get lipschitz constant of loss
    input_bounds = {}
    gradient_bounds = {}
    input_bound = model.X

    # Forward pass to assess maximum activation norms
    for layer in model.layers_forward_order():
        assert isinstance(layer, DPLayer)
        if model.cfg.run_eagerly:
            print(f"Layer {layer.name} input bound: {input_bound}")
        input_bounds[layer.name] = input_bound
        input_bound = layer.propagate_inputs(input_bound)

    if model.cfg.run_eagerly:
        print(f"Layer {layer.name} input bound: {input_bound}")

    gradient_bound = get_lip_constant_loss(model.cfg, input_bound)

    # Backward pass to compute gradient norm bounds and accumulate Lip constant
    for layer in model.layers_backward_order():
        assert isinstance(layer, DPLayer)
        layer_input_bound = input_bounds[layer.name]
        if layer.has_parameters():
            assert len(layer.trainable_variables) == 1
            var_name = layer.trainable_variables[0].name
            gradient_bounds[var_name] = layer.backpropagate_params(
                layer_input_bound, gradient_bound
            )
        gradient_bound = layer.backpropagate_inputs(layer_input_bound, gradient_bound)

    # Return gradient bounds
    return gradient_bounds


def get_noise_multiplier_coefs(model):
    dict = {}
    for layer in model.layers[::-1]:
        if isinstance(layer, DPLayer):
            if layer.has_parameters():
                assert len(layer.trainable_variables) == 1
                dict[layer.trainable_variables[0].name] = layer.nm_coef
    return dict


def local_noisify(model, gradient_bounds, trainable_vars, gradients):
    nm_coefs = get_noise_multiplier_coefs(model)
    noises = []
    for grad, var in zip(gradients, trainable_vars):
        if var.name in gradient_bounds.keys():
            stddev = (
                model.cfg.noise_multiplier
                * gradient_bounds[var.name]
                * nm_coefs[var.name]
            )
            noises.append(tf.random.normal(shape=grad.shape, stddev=stddev))
            if model.cfg.run_eagerly:
                upperboundgrad = gradient_bounds[var.name] * np.sqrt(model.cfg.batch_size)
                noise_msg = (
                    f"Adding noise of stddev : {stddev}"
                    f" to variable {var.name}"
                    f" of gradient norm upper bound {upperboundgrad}"
                    f" and effective norm {tf.norm(grad)}"
                )
                print(noise_msg)
        else:
            raise ValueError(f"Variable {var.name} not in gradient bounds.")

    noisy_grads = [g + n for g, n in zip(gradients, noises)]
    return noisy_grads


def global_noisify(model, gradient_bounds, trainable_vars, gradients):
    global_sensitivity = np.sqrt(
        sum([bound**2 for bound in gradient_bounds.values()])
    )
    stddev = model.cfg.noise_multiplier * global_sensitivity
    noises = [tf.random.normal(shape=g.shape, stddev=stddev) for g in gradients]
    if model.cfg.run_eagerly:
        for grad, var in zip(gradients, trainable_vars):
            upperboundgrad = gradient_bounds[var.name] * np.sqrt(model.cfg.batch_size)
            noise_msg = (
                f"Adding noise of stddev : {stddev}"
                f" to variable {var.name}"
                f" of gradient norm upper bound {upperboundgrad}"
                f" and effective norm {tf.norm(grad)}"
            )
            print(noise_msg)
    noisy_grads = [g + n for g, n in zip(gradients, noises)]
    return noisy_grads


class DP_Sequential(deel.lip.model.Sequential):
    """ "Model Class based on the DEEL Sequential model. Takes into account the architecture, only the following components are allowed :
    - Input
    - SpectralDense
    - SpectralConv2D
    - Flatten
    - ScaledL2GlobalPooling
    Args :
      Architecture: Sequential like input
      cfg: Contains the experiment config, must contain the batch size information and the chosen noise multiplier.
      X: The previously determined maximum norm among the training set.
    The model is then calibrated to verify (epsilon,delta)-DP guarantees by noisying the values of the gradients during the training step.
    Do not modify the config object after the model has been declared with config object cfg.
    Hypothesis: The model is 1-Lipschitz and Dense layers are Gradient Norm Preserving.
    """

    def __init__(self, *args, X, noisify_strategy, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = X
        self.cfg = cfg
        if noisify_strategy == "global":
            self.noisify_fun = global_noisify
        elif noisify_strategy == "local":
            self.noisify_fun = local_noisify
        else:
            raise TypeError(
                "Incorrect noisify_strategy argument during model initialisation."
            )

    def layers_forward_order(self):
        return self.layers

    def layers_backward_order(self):
        return self.layers[::-1]

    # Define the differentially private training step
    def train_step(self, data):
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # tf.cast(y_pred,dtype=y.dtype)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Get gradient bounds
        gradient_bounds = compute_gradient_bounds(model=self)
        noisy_gradients = self.noisify_fun(
            self, gradient_bounds, trainable_vars, gradients
        )
        # Each optimizer is a postprocessing of the already (epsilon,delta)-DP gradients
        self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update Metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Condense to verify |W|_2 = 1
        if self.cfg.condense:
            self.condense()
        return {m.name: m.result() for m in self.metrics}


class DP_Model(deel.lip.model.Model):
    """ "Model Class based on the DEEL Sequential model. Takes into account the architecture, only the following components are allowed :
    - Input
    - SpectralDense
    - SpectralConv2D
    - Flatten
    - ScaledL2GlobalPooling
    Args :
      Architecture: Sequential like input
      cfg: Contains the experiment config, must contain the batch size information and the chosen noise multiplier.
      X: The previously determined maximum norm among the training set.
    The model is then calibrated to verify (epsilon,delta)-DP guarantees by noisying the values of the gradients during the training step.
    Do not modify the config object after the model has been declared with config object cfg.
    Hypothesis: The model is 1-Lipschitz and Dense layers are Gradient Norm Preserving.
    """

    def __init__(self, dp_layers, *args, X, noisify_strategy, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_layers = dp_layers
        self.X = X
        self.cfg = cfg
        if noisify_strategy == "global":
            self.noisify_fun = global_noisify
        elif noisify_strategy == "local":
            self.noisify_fun = local_noisify
        else:
            raise TypeError(
                "Incorrect noisify_strategy argument during model initialisation."
            )

    def layers_forward_order(self):
        return self.dp_layers

    def layers_backward_order(self):
        return self.dp_layers[::-1]

    def call(self, inputs, *args, **kwarsg):
        x = inputs
        for layer in self.layers_forward_order():
            x = layer(x, *args, **kwarsg)
        return x

    # Define the differentially private training step
    def train_step(self, data):
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # tf.cast(y_pred,dtype=y.dtype)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Get gradient bounds
        gradient_bounds = compute_gradient_bounds(model=self)
        noisy_gradients = self.noisify_fun(
            self, gradient_bounds, trainable_vars, gradients
        )
        # Each optimizer is a postprocessing of the already (epsilon,delta)-DP gradients
        self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update Metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Condense to verify |W|_2 = 1
        if self.cfg.condense:
            self.condense()
        return {m.name: m.result() for m in self.metrics}
