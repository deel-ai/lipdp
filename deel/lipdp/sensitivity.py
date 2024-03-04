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

from deel.lipdp.model import get_eps_delta


def get_max_epochs(epsilon_max, model, epochs_max=1024, safe=True, atol=1e-2):
    """Return the maximum number of epochs to reach a given epsilon_max value.

    The computation of (epsilon, delta) is slow since it involves solving a minimization problem
    (mandatory to go from RDP accountant to DP results). Hence each call takes typically around 1s.
    This function is used to avoid calling get_eps_delta too many times be leveraging the fact that
    epsilon is a non-decreasing function of the number of epochs: we unlocks the dichotomy search.

    Hence the number of calls is typically log2(epochs_max) + 1.
    The maximum of epochs is set to 1024 by default to avoid too long computation times, even in high
    privacy regimes.

    Args:
        epsilon_max: The maximum value of epsilon we want to reach.
        model: The model used to compute the values of epsilon.
        epochs_max: The maximum number of epochs to reach epsilon_max. Defaults to 1024.
                    If None, the dichotomy search is used to find the upper bound.
        safe: If True, the dichotomy search returns the largest number of epochs such that epsilon <= epsilon_max.
              Otherwise, it returns the smallest number of epochs such that epsilon >= epsilon_max.
        atol: The absolute tolerance to panic on numerical inaccuracy. Defaults to 1e-2.

    Returns:
        The maximum number of epochs to reach epsilon_max. It may be zero if epsilon_max is too small.
    """
    steps_per_epoch = model.dataset_metadata.nb_steps_per_epochs

    def fun(epoch):
        if epoch == 0:
            epsilon = 0
        else:
            epsilon, _ = get_eps_delta(model, epoch)
        return epsilon

    # dichotomy search on the number of epochs.
    if epochs_max is None:
        epochs_max = 512
        epsilon = 0
        while epsilon < epsilon_max:
            epochs_max *= 2
            epsilon = fun(epochs_max)
            print(f"epochs_max = {epochs_max} at epsilon = {epsilon}")

    epochs_min = 0

    while epochs_max - epochs_min > 1:
        epoch = (epochs_max + epochs_min) // 2
        epsilon = fun(epoch)
        if epsilon < epsilon_max:
            epochs_min = epoch
        else:
            epochs_max = epoch
        print(
            f"epoch bounds = {epochs_min, epochs_max} and epsilon = {epsilon} at epoch {epoch}"
        )

    if safe:
        last_epsilon = fun(epochs_min)
        error = last_epsilon - epsilon_max
        if error <= 0:
            return epochs_min
        elif error < atol:
            # This branch should never be taken if fun is a non-decreasing function of the number of epochs.
            # fun is mathematcally non-decreasing, but numerical inaccuracy can lead to this case.
            print(f"Numerical inaccuracy with error {error:.7f} in the dichotomy search: using a conservative value.")
            return epochs_min - 1
        else:
            assert False, f"Numerical inaccuracy with error {error:.7f}>{atol:.3f} in the dichotomy search."

    return epochs_max


def gradient_norm_check(upper_bounds, model, examples):
    """Verifies that the values of per-sample gradients on a layer never exceede a value
    determined by the theoretical work.

    Args :
        upper_bounds: maximum gradient bounds for each layer (dictionnary of 'layers name ': 'bounds' pairs).
        model: The model containing the layers we are interested in. Layers must only have one trainable variable.
        examples: a batch of examples to test on.  
    Returns :
        Boolean value. True corresponds to upper bound has been validated.
    """
    activations = examples
    var_seen = set()
    for layer in model.layers:
        post_activations = layer(activations, training=True)
        assert len(layer.trainable_variables) < 2
        if len(layer.trainable_variables) == 1:
            assert len(layer.trainable_variables) == 1
            train_var = layer.trainable_variables[0]
            var_name = layer.trainable_variables[0].name
            var_seen.add(var_name)
            bound = upper_bounds[var_name]
            check_layer_gradient_norm(bound, layer, activations)
        activations = post_activations
    for var_name in upper_bounds:
        assert var_name in var_seen


def check_layer_gradient_norm(S, layer, activations):
    trainable_vars = layer.trainable_variables[0]
    with tf.GradientTape() as tape:        
        y_pred = layer(activations, training=True)
        flat_pred = tf.reshape(y_pred, (y_pred.shape[0], -1))
    jacobians = tape.jacobian(flat_pred, trainable_vars)
    assert jacobians.shape[0] == activations.shape[0]
    assert jacobians.shape[1] == np.prod(y_pred.shape[1:])
    assert np.prod(jacobians.shape[2:]) == np.prod(trainable_vars.shape)
    jacobians = tf.reshape(
        jacobians,
        (y_pred.shape[0], -1, np.prod(trainable_vars.shape)),
        name="Reshaped_Gradient",
    )
    J_sigma = tf.linalg.svd(jacobians, full_matrices=False, compute_uv=False, name=None)
    J_2norm = tf.reduce_max(J_sigma, axis=-1)
    J_2norm = tf.reduce_max(J_2norm).numpy()
    atol = 1e-5
    return J_2norm < S+atol
