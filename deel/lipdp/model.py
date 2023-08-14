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
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from autodp import mechanism_zoo
from autodp import transformer_zoo
from autodp.autodp_core import Mechanism
from tensorflow import keras

import deel
from deel.lipdp.layers import DP_ClipGradient
from deel.lipdp.layers import DPLayer
from deel.lipdp.pipeline import DatasetMetadata


def clipsum(norms, C):
    """
    Computes the sum of individually clipped elements from the given list or tensor.

    Args:
        norms (list or tensor): A list or tensor containing the elements to be clipped and summed.
        C (float): A clipping constant used to clip the elements.

    Returns:
        float: The sum of the clipped elements.

    Example:
        >>> norms = [1.3, 2.7, 7.5]
        >>> C = 3.0
        >>> clipsum(norms, C)
        7.0
    """
    norms = tf.cast(norms, dtype=tf.float32)
    C = tf.constant([C])
    C = tf.cast(C, dtype=tf.float32)
    return tf.math.reduce_sum(tf.math.minimum(norms, C))


def diff_query(norms, lower, upper, n_points=1000):
    """
    Computes the difference between two sums of clipped elements with two different clipping constants
    on a range of n_points between the lower and upper values.

    Args:
        norms (list or tensor): A list or tensor of values to be clipped and summed.
        lower (float or int): The lower bound of the search range.
        upper (float or int): The upper bound of the search range.
        n_points (int): The number of points between the lower and upper bound.

    Returns:
        alpha (float): The sensitivity of the differentiation query, calculated as (upper - lower) / (n_points - 1).
        points (list): The list of points iterated on between the lower and upper bound.
        queries (float): The values of the difference query over the points range.

    """
    points = np.linspace(lower, upper, num=n_points)
    alpha = (upper - lower) / (n_points - 1)
    queries = []
    for p in points:
        query = clipsum(norms, p) - clipsum(norms, p + alpha)
        queries.append(query)
    return alpha, points, queries


def above_treshold(queries, sensitivity, T, epsilon):
    """
    SVT inspired algorithm inspired from https://programming-dp.com/ch10.html. Computes
    the index for which the differentiation query of the queries list converges above a
    treshold T. This computation is epsilon-DP.

    Args :
        queries (list or tensor) : list of the values of the difference query.
        sensitivity (float) : sensitivity of the difference computation query.
        T (float) : value of the treshold.
        epsilon (float) : chosen epsilon guarantee on the query.

    Returns :
        ids (int) : the index corresponding the epsilon-DP estimated optimal clipping constant.

    """
    T_hat = T + np.random.laplace(loc=0, scale=2 * sensitivity / epsilon)
    for idx, q in enumerate(queries):
        nu_i = np.random.laplace(loc=0, scale=4 * sensitivity / epsilon)
        if q + nu_i >= T_hat:
            return idx
    return random.randint(0, len(queries) - 1)


class AdaptiveLossGradientClipping(keras.callbacks.Callback):
    """Updates the clipping value of the last layer of the model.

    This callback privately updates the clipping value if the last layer
    of the model is a DP_ClipGradient layer with mode = "dynamic_svt".

    Attributes :
        ds_train : a tensorflow dataset object.
    """

    def __init__(self, ds_train=None):
        self.ds_train = ds_train

    def on_train_begin(self, logs=None):
        # Check that callback is called on a model with a clipping layer at the end
        assert isinstance(self.model.layers_backward_order()[0], DP_ClipGradient)
        print("On train begin : ")
        self.model.layers_backward_order()[0].initial_value = tf.convert_to_tensor(
            self.model.loss.get_L(), dtype=tf.float32
        )
        print(
            "Initial value is now equal to lipschitz constant of loss: ",
            self.model.layers_backward_order()[0].initial_value,
        )
        self.model.layers_backward_order()[0].clip_value.assign(
            tf.convert_to_tensor(self.model.loss.get_L(), dtype=tf.float32)
        )
        return

    def on_epoch_end(self, epoch, logs={}):
        last_layer = self.model.layers_backward_order()[0]
        assert isinstance(last_layer, DP_ClipGradient)
        # print("Patience : ", epoch % last_layer.patience)
        if last_layer.mode == "fixed":
            raise TypeError(
                "Fixed mode for last layer is incompatible with this callback"
            )
        if epoch % last_layer.patience == 0:
            epsilon = last_layer.epsilon
            list_norms = self.get_gradloss()
            alpha, points, queries = diff_query(
                list_norms, lower=0, upper=self.model.loss.get_L()
            )
            T = queries[0] * 0.1
            updated_clip_value = points[
                above_treshold(queries, sensitivity=alpha, T=T, epsilon=epsilon)
            ]
            print("updated_clip_value : ", updated_clip_value)
            self.model.layers_backward_order()[0].clip_value.assign(updated_clip_value)
            return

    def get_gradloss(self):
        batch = next(iter(self.ds_train.take(1)))
        imgs, labels = batch
        self.model.loss.reduction = tf.keras.losses.Reduction.NONE
        predictions = self.model(imgs)
        with tf.GradientTape() as tape:
            tape.watch(predictions)
            loss_value = self.model.compiled_loss(labels, predictions)
        self.model.loss.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        grad_loss = tape.gradient(loss_value, predictions)
        norms = tf.norm(grad_loss, axis=-1)
        return norms


@dataclass
class DPParameters:
    """Parameters used to set the dp training.

    Attributes:
        noisify_strategy (str): either 'local' or 'global'.
        noise_multiplier (float): noise multiplier.
        delta (float): delta parameter for DP.
    """

    noisify_strategy: str
    noise_multiplier: float
    delta: float


class DPGD_Mechanism(Mechanism):
    """DPGD Mechanism.

    Args:
        mode (str): kind of mechanism to use. Either 'global' or 'local'.
        prob (float): probability of subsampling.
        noise_multipliers (float, or list of floats): single scalar when mode == 'global', list of scalars when mode == 'local'.
        num_grad_steps (int): number of gradient steps.
        delta (float): delta parameter for DP.
        dynamic_clipping (optional, dict): dictionary of parameters for dynamic clipping with keys:
            epsilon (float): epsilon parameter for SVT algorithm.
            num_updates (int): patience parameter for SVT algorithm.
    """

    def __init__(
        self,
        mode,
        prob,
        noise_multipliers,
        num_grad_steps,
        delta,
        dynamic_clipping=None,
        name="DPGD_Mechanism",
    ):
        # Init
        Mechanism.__init__(self)
        self.name = name
        self.params = {
            "prob": prob,
            "noise_multipliers": noise_multipliers,
            "num_grad_steps": num_grad_steps,
            "delta": delta,
            "dynamic_clipping": dynamic_clipping,
        }

        if mode == "global":
            model_mech = mechanism_zoo.GaussianMechanism(sigma=noise_multipliers)
            assert model_mech.neighboring == "add_remove"
        elif mode == "local":
            layer_mechanisms = []

            for sigma in noise_multipliers:
                mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
                assert model_mech.neighboring == "add_remove"
                layer_mechanisms.append(mech)

            # Accountant composition on layers
            compose_gaussians = transformer_zoo.ComposeGaussian()
            model_mech = compose_gaussians(
                layer_mechanisms, [1] * len(noise_multipliers)
            )
        else:
            raise ValueError("Unknown kind of mechanism")

        subsample = transformer_zoo.AmplificationBySampling()
        SubsampledModelGaussian_mech = subsample(
            # improved_bound_flag can be set to True for Gaussian mechanisms (see autodp documentation).
            model_mech,
            prob,
            improved_bound_flag=True,
        )
        compose = transformer_zoo.Composition()

        if dynamic_clipping is None or dynamic_clipping["mode"] == "fixed":
            global_mech = compose([SubsampledModelGaussian_mech], [num_grad_steps])
        elif dynamic_clipping["mode"] == "dynamic_svt":
            DynamicClippingMech = mechanism_zoo.PureDP_Mechanism(
                eps=dynamic_clipping["epsilon"], name="SVT"
            )
            global_mech = compose(
                [SubsampledModelGaussian_mech, DynamicClippingMech],
                [num_grad_steps, dynamic_clipping["num_updates"]],
            )

        assert global_mech.neighboring in ["add_remove", "add_only", "remove_only"]

        # Get relevant information
        self.epsilon = global_mech.get_approxDP(delta=delta)
        self.delta = delta

        # Propagate updates
        rdp_global = global_mech.RenyiDP
        self.propagate_updates(rdp_global, type_of_update="RDP")


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
        self.log_fn({"epsilon": epsilon, "delta": delta, "epoch": epoch + 1})


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

    last_layer = model.layers_backward_order()[0]
    dynamic_clipping = None
    if isinstance(last_layer, deel.lipdp.layers.DP_ClipGradient):
        dynamic_clipping = {}
        dynamic_clipping["mode"] = last_layer.mode
        dynamic_clipping["epsilon"] = last_layer.epsilon
        dynamic_clipping["num_updates"] = epochs // last_layer.patience

    if model.dp_parameters.noisify_strategy == "local":
        nm_coefs = get_noise_multiplier_coefs(model)
        noise_multipliers = [
            model.dp_parameters.noise_multiplier * coef for coef in nm_coefs.values()
        ]
        mode = "local"
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
            TypeError: when the dp_parameters.noisify_strategy is not one of "local" or "global"
        """
        super().__init__(*args, **kwargs)
        self.dp_parameters = dp_parameters
        self.dataset_metadata = dataset_metadata
        self.debug = debug
        if self.dp_parameters.noisify_strategy == "global":
            self.noisify_fun = global_noisify
        elif self.dp_parameters.noisify_strategy == "local":
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
            TypeError: when the dp_parameters.noisify_strategy is not one of "local" or "global"
        """
        super().__init__(*args, **kwargs)
        self.dp_layers = dp_layers
        self.dp_parameters = dp_parameters
        self.dataset_metadata = dataset_metadata
        self.debug = debug
        if self.dp_parameters.noisify_strategy == "global":
            self.noisify_fun = global_noisify
        elif self.dp_parameters.noisify_strategy == "local":
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
        self.condense()
        return {m.name: m.result() for m in self.metrics}
