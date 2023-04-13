import numpy as np
import tensorflow as tf


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
