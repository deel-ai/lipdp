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
        niter = (epoch + 1) * (self.model.cfg.N // self.model.cfg.batch_size)
        epsilon, delta = get_eps_delta(model=self.model, niter=niter)
        print(f"\n {(epsilon,delta)}-DP guarantees for epoch {epoch+1} \n")
        wandb.log({"epsilon": epsilon})


class Global_DPGD_Mechanism(Mechanism):
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
    for layer in model.layers:
        if isinstance(layer, DPLayer):
            if layer.has_parameters():
                assert len(layer.trainable_variables) == 1
                dict_coefs[layer.trainable_variables[0].name] = layer.nm_coef
    return dict_coefs


def get_eps_delta(model, niter):
    nm_coefs = get_nm_coefs(model)
    prob = model.cfg.batch_size / model.cfg.N
    # nm_coefs.values is seamingly in the right order :
    sigmas = [model.cfg.noise_multiplier * coef for coef in nm_coefs.values()]
    mech = Global_DPGD_Mechanism(
        prob=prob, sigmas=sigmas, delta=model.cfg.delta, niter=niter
    )
    return mech.epsilon, mech.delta


def compute_gradient_bounds(model):
    # Initialisation, get lipschitz constant of loss
    dict_H, dict_K = {}, {}
    H = model.X
    K_loss = get_lip_constant_loss(model.cfg)

    # Forward pass to assess maximum activation norms
    for layer in model.layers:
        if isinstance(layer, DPLayer):
            if layer.has_parameters():
                assert len(layer.trainable_variables) == 1
                dict_H[layer.name] = layer.get_DP_LipCoef_inputs(H)
                H = layer.get_DP_LipCoef_inputs(H)
            else:
                H = layer.get_DP_LipCoef_inputs(H)
        # else :
        #   print(f"WARNING : {layer.name} is not defined as a DP layer")

    # print("dict_activation_norms", dict_H)

    L_tilde = K_loss
    # Backward pass to compute gradient norm bounds and accumulate Lip constant
    for layer in model.layers[::-1]:
        if isinstance(layer, DPLayer):
            if layer.has_parameters():
                assert len(layer.trainable_variables) == 1
                dict_K[layer.trainable_variables[0].name] = layer.get_DP_LipCoef_params(
                    L_tilde * dict_H[layer.name]
                )
                # print(f"For layer {layer.name} gradient norm bound is : {layer.get_DP_LipCoef_params(L_tilde * dict_H[layer.name])}")
                # WARNING : UNCHECKED
                L_tilde = L_tilde * layer.get_DP_LipCoef_inputs(1)
            else:
                # print(f"For layer {layer.name} gradient L_tilde coefficient is updated to : {L_tilde * layer.get_DP_LipCoef_inputs(1)}")
                # WARNING : UNCHECKED
                L_tilde = L_tilde * layer.get_DP_LipCoef_inputs(1)

    # Return gradient bounds
    return dict_K


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
        # PB for Layer Centering appears as var
        if var.name in gradient_bounds.keys():
            stddev = (
                model.cfg.noise_multiplier
                * gradient_bounds[var.name]
                * nm_coefs[var.name]
                / model.cfg.batch_size
            )
            noises.append(tf.random.normal(shape=grad.shape, stddev=stddev))
            if model.cfg.run_eagerly:
                print(
                    f"Adding noise of stddev : {stddev} to variable {var.name} of bound {gradient_bounds[var.name]} and effective value {tf.norm(grad)}"
                )
        else:
            noises.append(tf.zeros(shape=grad.shape))

    noisy_grads = [g + n for g, n in zip(gradients, noises)]
    return noisy_grads


def global_noisify(model, gradient_bounds, trainable_vars, gradients):
    global_bound = np.sqrt(sum([bound**2 for bound in gradient_bounds.values()]))
    stddev = model.cfg.noise_multiplier * global_bound / model.cfg.batch_size
    noises = [tf.random.normal(shape=g.shape, stddev=stddev) for g in gradients]
    noisy_grads = [g + n for g, n in zip(gradients, noises)]
    return noisy_grads


class DP_LipNet(deel.lip.model.Sequential):
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


class DP_LipNet_Model(deel.lip.model.Model):
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
