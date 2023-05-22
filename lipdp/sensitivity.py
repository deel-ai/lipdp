# -*- coding: utf-8 -*-
# Copyright anonymized et anonymized - All
# rights reserved. anonymized is a research program operated by anonymized, anonymized,
# anonymized and anonymized - https://www.anonymized.ai/
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

from lipdp.model import get_eps_delta


def get_max_epochs(epsilon_max, model, epochs_max=1024):
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

    Returns:
        The maximum number of epochs to reach epsilon_max."""
    steps_per_epoch = model.dataset_metadata.nb_steps_per_epochs

    def fun(epoch):
        if epoch == 0:
            epsilon = 0
        else:
            epoch = round(epoch)
            niter = (epoch + 1) * steps_per_epoch
            epsilon, _ = get_eps_delta(model, niter)
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
        epoch = (epochs_max + epochs_min) / 2
        epsilon = fun(epoch)
        if epsilon < epsilon_max:
            epochs_min = epoch
        else:
            epochs_max = epoch
        print(
            f"epoch bounds = {epochs_min, epochs_max} and epsilon = {epsilon} at epoch {epoch}"
        )

    return int(round(epoch))


def gradient_norm_check(K_list, model, examples):
    """
    Verifies that the values of per-sample gradients on a layer never exceede a theoretical value
    determined by our theoretical work.
    Args :
        Klist: The list of theoretical upper bounds we have identified for each layer and want to
        put to the test.
        model: The model containing the layers we are interested in. Layers must only have one trainable variable.
        Model must have a given input_shape or has to be built.
        examples: Relevant examples. Inputting the whole training set might prove very costly to check element wise Jacobians.
    Returns :
        Boolean value. True corresponds to upper bound has been validated.
    """
    image_axes = tuple(range(1, examples.ndim))
    example_norms = tf.math.reduce_euclidean_norm(examples, axis=image_axes)
    X_max = tf.reduce_max(example_norms).numpy()
    upper_bounds = np.array(K_list) * X_max
    assert len(model.layers) == len(upper_bounds)
    for layer, bound in zip(model.layers, upper_bounds):
        assert check_layer_gradient_norm(bound, layer, examples)


def check_layer_gradient_norm(S, layer, examples):
    l_model = tf.keras.Sequential([layer])
    if not l_model.trainable_variables:
        print("Not a trainable layer assuming gradient norm < |x|")
    assert len(l_model.trainable_variables) == 1
    with tf.GradientTape() as tape:
        y_pred = l_model(examples, training=True)
    trainable_vars = l_model.trainable_variables[0]
    jacobian = tape.jacobian(y_pred, trainable_vars)
    jacobian = tf.reshape(
        jacobian,
        (y_pred.shape[0], -1, np.prod(trainable_vars.shape)),
        name="Reshaped_Gradient",
    )
    J_sigma = tf.linalg.svd(jacobian, full_matrices=False, compute_uv=False, name=None)
    J_2norm = tf.reduce_max(J_sigma, axis=-1)
    J_2norm = tf.reduce_max(J_2norm).numpy()
    return J_2norm < S
