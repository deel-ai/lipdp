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
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

import deel
from deel.lipdp.accounting import DPGD_Mechanism
from deel.lipdp.layers import DPLayer
from deel.lipdp.pipeline import DatasetMetadata


@dataclass
class DPParameters:
    """Parameters used to set the dp training.

    Attributes:
        noisify_strategy (str): either "per-layer" or "global".
        noise_multiplier (float): noise multiplier.
        delta (float): delta parameter for DP.
    """

    noisify_strategy: str
    noise_multiplier: float
    delta: float


class DP_Accountant(keras.callbacks.Callback):
    """Callback to compute the DP guarantees at the end of each epoch.

    Note: wandb is not a strict requirement for this callback to work, logging is also supported.

    Attributes:
        log_fn: log function to use. Takes a dictionary of (key, value) pairs as input.
                if 'wandb', use wandb.log.
                if 'logging', use logging.info.
                if 'all', use both wandb.log and logging.info.
    """

    def __init__(self, log_fn="all"):
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
        self.log_fn = log_fn

    def on_epoch_end(self, epoch, logs=None):
        epsilon, delta = get_eps_delta(model=self.model, epochs=epoch + 1)
        print(f"\n {(epsilon,delta)}-DP guarantees for epoch {epoch+1} \n")

        # plot epoch at the same time as epsilon and delta to ease comparison of plots in wandb API.
        to_log = {
            "epsilon": epsilon,
            "delta": delta,
            "epoch": epoch + 1,
        }

        last_layer = self.model.layers_backward_order()[0]
        if isinstance(last_layer, deel.lipdp.layers.DP_ClipGradient):
            clipping_value = float(last_layer.clip_value.numpy())
            to_log["clipping_value"] = clipping_value

        self.log_fn(to_log)


def get_eps_delta(model, epochs):
    """Compute the (epsilon, delta)-DP guarantees of the model.

    Args:
        model: model to train.
        epochs: number of epochs elapsed.

    Returns:
        epsilon: epsilon parameter of the (epsilon, delta)-DP guarantee.
        delta: delta parameter of the (epsilon, delta)-DP guarantee.
    """
    num_grad_steps = epochs * model.dataset_metadata.nb_steps_per_epochs

    prob = model.dataset_metadata.batch_size / model.dataset_metadata.nb_samples_train

    # Dynamic clipping might be used.
    last_layer = model.layers_backward_order()[0]
    dynamic_clipping = {"mode": "fixed"}
    if isinstance(last_layer, deel.lipdp.layers.DP_ClipGradient):
        dynamic_clipping.update(last_layer._dynamic_dp_dict)  # copy dict
        if "patience" in dynamic_clipping:
            dynamic_clipping["num_updates"] = epochs // dynamic_clipping["patience"]

    if model.dp_parameters.noisify_strategy == "per-layer":
        nm_coefs = get_noise_multiplier_coefs(model)
        noise_multipliers = [
            model.dp_parameters.noise_multiplier * coef for coef in nm_coefs.values()
        ]
        mode = "per-layer"
    elif model.dp_parameters.noisify_strategy == "global":
        noise_multipliers = model.dp_parameters.noise_multiplier
        mode = "global"

    mech = DPGD_Mechanism(
        mode=mode,
        prob=prob,
        noise_multipliers=noise_multipliers,
        num_grad_steps=num_grad_steps,
        delta=model.dp_parameters.delta,
        dynamic_clipping=dynamic_clipping,
    )

    return mech.epsilon, mech.delta


def get_noise_multiplier_coefs(model):
    """Get the noise multiplier coefficients of the model.

    Args:
        model: model to train.

    Returns:
        dict_coefs: dictionary of noise multiplier coefficients.
                    The order of the coefficients is the same as
                    the order of the layers returned by model.layers_forward_order().
    """
    dict_coefs = {}
    for (
        layer
    ) in (
        model.layers_forward_order()
    ):  # remark: insertion order is preserved in Python 3.7+
        assert isinstance(layer, DPLayer)
        if layer.has_parameters():
            assert len(layer.trainable_variables) == 1
            dict_coefs[layer.trainable_variables[0].name] = layer.nm_coef
    return dict_coefs


def compute_gradient_bounds(model):
    """Compute the gradient norm bounds of the model.

    Args:
        model: model to train.

    Returns:
        gradient_bounds: dictionary of gradient norm bounds with (key, value) pairs (layer_name, gradient_bound).
                         The order of the bounds is the same as
                         the order of the layers returned by model.layers_backward_order().
    """
    # Initialisation, get lipschitz constant of loss
    input_bounds = {}
    gradient_bounds = {}
    input_bound = None  # Unknown at the start.

    # Forward pass to assess maximum activation norms
    for layer in model.layers_forward_order():
        assert isinstance(layer, DPLayer)
        if model.debug:
            print(f"Layer {layer.name} input bound: {input_bound}")
        input_bounds[layer.name] = input_bound
        input_bound = layer.propagate_inputs(input_bound)

    if model.debug:
        print(f"Layer {layer.name} input bound: {input_bound}")

    # since we aggregate using SUM_OVER_BATCH
    gradient_bound = tf.convert_to_tensor(model.loss.get_L()) / tf.convert_to_tensor(
        model.dataset_metadata.batch_size, dtype=tf.float32
    )

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


def local_noisify(model, gradient_bounds, trainable_vars, gradients):
    """Add noise to gradients of trainable variables.

    Remark: this yields tighter bounds than global_noisify.

    Args:
        model: model to train.
        gradient_bounds: dictionary of gradient norm upper bounds. Keys are trainable variables names.
        trainable_vars: list of trainable variables. Same order as gradients.
        gradients: list of gradients. Same order as trainable_vars.

    Returns:
        list of noisy gradients. Same order as trainable_vars.
    """
    nm_coefs = get_noise_multiplier_coefs(model)
    noises = []
    for grad, var in zip(gradients, trainable_vars):
        if var.name in gradient_bounds.keys():
            # no factor-2 : use add_remove definition of DP
            stddev = (
                model.dp_parameters.noise_multiplier
                * gradient_bounds[var.name]
                * nm_coefs[var.name]
            )
            noises.append(tf.random.normal(shape=tf.shape(grad), stddev=stddev))
            if model.debug:
                upperboundgrad = (
                    gradient_bounds[var.name] * model.dataset_metadata.batch_size
                )
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
    """Add noise to gradients.

    Remark: a single global noise is added to all gradients, based on the global sensitivity.
            This is the default behaviour of the original DPGD algorithm.
            This may yield looser privacy bounds than local noisify.

    Args:
        model: model to train.
        gradient_bounds: dictionary of gradient norm upper bounds. The keys are the names of the trainable variables.
        trainable_vars: list of trainable variables. The list is in the same order as gradients.
        gradients: list of gradients to add noise to. The list is in the same order as trainable_vars.

    Returns:
        noisy_grads: list of noisy gradients. The list is in the same order as trainable_vars.
    """
    global_sensitivity = tf.math.sqrt(
        tf.math.reduce_sum([bound**2 for bound in gradient_bounds.values()])
    )
    # no factor-2 : use add_remove definition of DP.
    stddev = model.dp_parameters.noise_multiplier * global_sensitivity
    noises = [tf.random.normal(shape=tf.shape(g), stddev=stddev) for g in gradients]
    if model.debug:
        for grad, var in zip(gradients, trainable_vars):
            upperboundgrad = (
                gradient_bounds[var.name] * model.dataset_metadata.batch_size
            )
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
    def __init__(
        self,
        *args,
        dp_parameters: DPParameters,
        dataset_metadata: DatasetMetadata,
        debug: bool = False,
        **kwargs,
    ):
        """Model Class based on the DEEL Sequential model. Only layer from the lipdp.layers module are allowed since
        the framework assume 1 lipschitz layers.

        Args:
            dp_parameters (DPParameters): parameters used to set the dp procedure.
            dataset_metadata (DatasetMetadata): information about the dataset. Must contain: the input shape, number
                of training samples, the input bound, number of batches in the dataset and the batch size.
            debug (bool, optional): when true print additionnal debug informations (must be in eager mode). Defaults to False.

        Note:
            The model is then calibrated to verify (epsilon,delta)-DP guarantees by noisying the values of the gradients during the training step.
            DP accounting is done with the associated Callback.

        Raises:
            TypeError: when the dp_parameters.noisify_strategy is not one of "per-layer" or "global"
        """
        super().__init__(*args, **kwargs)
        self.dp_parameters = dp_parameters
        self.dataset_metadata = dataset_metadata
        self.debug = debug
        if self.dp_parameters.noisify_strategy == "global":
            self.noisify_fun = global_noisify
        elif self.dp_parameters.noisify_strategy == "per-layer":
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
        self.condense()
        return {m.name: m.result() for m in self.metrics}


class DP_Model(deel.lip.model.Model):
    def __init__(
        self,
        dp_layers,
        *args,
        dp_parameters: DPParameters,
        dataset_metadata: DatasetMetadata,
        debug: bool = False,
        **kwargs,
    ):
        """Model Class based on the DEEL Sequential model. Only layer from the lipdp.layers module are allowed since
        the framework assume 1 lipschitz layers.

        Args:
            dp_layers: the list of layers to use ( as done in sequential ) but here we can leverage
                the fact that layers may have multiple inputs/outputs.
            dp_parameters (DPParameters): parameters used to set the dp procedure.
            dataset_metadata (DatasetMetadata): information about the dataset. Must contain: the input shape, number
                of training samples, the input bound, number of batches in the dataset and the batch size.
            debug (bool, optional): when true print additionnal debug informations (must be in eager mode). Defaults to False.

        Note:
            The model is then calibrated to verify (epsilon,delta)-DP guarantees by noisying the values of the gradients during the training step.
            DP accounting is done with the associated Callback.

        Raises:
            TypeError: when the dp_parameters.noisify_strategy is not one of "per-layer" or "global"
        """
        super().__init__(*args, **kwargs)
        self.dp_layers = dp_layers
        self.dp_parameters = dp_parameters
        self.dataset_metadata = dataset_metadata
        self.debug = debug
        if self.dp_parameters.noisify_strategy == "global":
            self.noisify_fun = global_noisify
        elif self.dp_parameters.noisify_strategy == "per-layer":
            self.noisify_fun = local_noisify
        else:
            raise TypeError(
                "Incorrect noisify_strategy argument during model initialisation."
            )

    def layers_forward_order(self):
        return self.dp_layers

    def layers_backward_order(self):
        return self.dp_layers[::-1]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers_forward_order():
            x = layer(x, *args, **kwargs)
        return x

    def signal_to_noise_elementwise(self, data):
        """Compute the signal to noise ratio of the model.

        Args:
            data: a tuple (x,y) of a batch of data.

        Returns:
            ratios: dictionary of signal to noise ratios. Keys are trainable variables names.
            norms: dictionary of gradient norms. Keys are trainable variables names.
            bounds: dictionary of gradient norm bounds. Keys are trainable variables names.
        """
        import tqdm

        examples, labels = data

        trainable_vars = self.trainable_variables
        names = [v.name for v in trainable_vars]

        bounds = compute_gradient_bounds(model=self)
        batch_size = self.dataset_metadata.batch_size
        bounds = {name: bound * batch_size for name, bound in bounds.items()}

        norms = {name: [] for name in names}
        ratios = {name: [] for name in names}
        total = len(examples)
        for x, y in tqdm.tqdm(zip(examples, labels), total=total):
            with tf.GradientTape() as tape:
                x = tf.expand_dims(x, axis=0)
                y = tf.expand_dims(y, axis=0)
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            gradient_element = tape.gradient(loss, self.trainable_variables)
            norms_element = [tf.linalg.norm(g, axis=None) for g in gradient_element]
            norms_element = {name: norm for name, norm in zip(names, norms_element)}
            for name in names:
                norms[name].append(norms_element[name].numpy())

            ratios_element = {}
            for name in names:
                ratios_element[name] = norms_element[name] / bounds[name]
            for name in names:
                ratios[name].append(ratios_element[name])

        ratios = {name: np.stack(ratios[name]) for name in names}
        norms = {name: np.stack(norms[name]) for name in names}

        return ratios, norms, bounds

    def signal_to_noise_average(self, data):
        """Compute the signal to noise ratio of the model.

        Args:
            data: a tuple (x,y) of a batch of data. The batch size must be equal to the one of the dataset.

        Returns:
            ratios: dictionary of signal to noise ratios. Keys are trainable variables names.
            norms: dictionary of gradient norms. Keys are trainable variables names.
            bounds: dictionary of gradient norm bounds. Keys are trainable variables names.
        """
        x, y = data

        assert (
            x.shape[0] == self.dataset_metadata.batch_size
        ), "Batch size must be equal to the one of the dataset"

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # tf.cast(y_pred,dtype=y.dtype)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # gradient norms
        norms = [tf.linalg.norm(g, axis=None) for g in gradients]
        names = [v.name for v in trainable_vars]
        norms = {name: norm for name, norm in zip(names, norms)}

        # Get gradient bounds
        bounds = compute_gradient_bounds(model=self)
        batch_size = self.dataset_metadata.batch_size
        bounds = {name: (bound * batch_size) for name, bound in bounds.items()}

        ratios = {}
        for name in names:
            ratios[name] = norms[name] / bounds[name]
        return ratios, norms, bounds

    # Define the differentially private training step
    def train_step(self, data):
        """Train step of the model with DP guarantees.

        Args:
            data: a tuple (x,y) of a batch of data.

        Returns:
            metrics: dictionary of metrics.
        """
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
        # Each optimizer is a postprocessing of private gradients
        self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))

        # Update Metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Condense to ensure Lipschitz constraints |W|_2 = 1
        self.condense()
        return {m.name: m.result() for m in self.metrics}
