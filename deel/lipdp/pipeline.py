import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical


def load_data_cifar(cfg):
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = cifar10.load_data()
    # Normalize
    x_train = x_train / 255
    x_test = x_test / 255
    if cfg.representation == "HSV":
        x_train, x_test = tf.image.rgb_to_hsv(x_train), tf.image.rgb_to_hsv(x_test)
    if cfg.representation == "YIQ":
        x_train, x_test = tf.image.rgb_to_yiq(x_train), tf.image.rgb_to_yiq(x_test)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get theoretical L2 norm upper bound
    theoretical_upper_bound = np.sqrt(np.prod(x_train[0].shape))
    upper_bound = theoretical_upper_bound * cfg.input_clipping
    # Clip training and testing set imgs
    x_train = tf.clip_by_norm(x_train, upper_bound, axes=list(range(1, x_train.ndim)))
    x_test = tf.clip_by_norm(x_test, upper_bound, axes=list(range(1, x_train.ndim)))
    return x_train, x_test, y_train, y_test, upper_bound


def load_data_mnist(cfg):
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = mnist.load_data()
    # Normalize
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get theoretical L2 norm upper bound
    theoretical_upper_bound = np.sqrt(np.prod(x_train[0].shape))
    upper_bound = theoretical_upper_bound * cfg.input_clipping
    # Clip training and testing set imgs
    x_train = tf.clip_by_norm(x_train, upper_bound, axes=list(range(1, x_train.ndim)))
    x_test = tf.clip_by_norm(x_test, upper_bound, axes=list(range(1, x_train.ndim)))
    return x_train, x_test, y_train, y_test, upper_bound


def load_data_fashion_mnist(cfg):
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = fashion_mnist.load_data()
    # Normalize
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get theoretical L2 norm upper bound
    theoretical_upper_bound = np.sqrt(np.prod(x_train[0].shape))
    upper_bound = theoretical_upper_bound * cfg.input_clipping
    # Clip training and testing set imgs
    x_train = tf.clip_by_norm(x_train, upper_bound, axes=list(range(1, x_train.ndim)))
    x_test = tf.clip_by_norm(x_test, upper_bound, axes=list(range(1, x_train.ndim)))
    return x_train, x_test, y_train, y_test, upper_bound
