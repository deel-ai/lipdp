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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_data_cifar(cfg):
    """
    Loads the CIFAR10 dataset and outputs a tuple containing x_train,x_test,y_train,y_test,upper_bound. Four image representation
    systems are allowed "RGB", "HSV", "YIQ", "YUV". The upper_bound is computed on the clipped dataset of chosen representation.

    Args :
        cfg : Config object containing the chosen representation allow with the input_clipping factor.

    Returns :
        x_train (training dataset)
        y_train (training labels)
        x_test (testing dataset)
        y_test (testing labels)
        upper_bound (float) : Value of the maximum norm of transformed and clipped training dataset.
    """
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = cifar10.load_data()
    # Normalize
    x_train = x_train / 255
    x_test = x_test / 255
    if cfg.representation == "RGB":
        pass
    elif cfg.representation == "HSV":
        x_train, x_test = tf.image.rgb_to_hsv(x_train), tf.image.rgb_to_hsv(x_test)
    elif cfg.representation == "YIQ":
        x_train, x_test = tf.image.rgb_to_yiq(x_train), tf.image.rgb_to_yiq(x_test)
    elif cfg.representation == "YUV":
        x_train, x_test = tf.image.rgb_to_yuv(x_train), tf.image.rgb_to_yuv(x_test)
    else:
        raise ValueError("Incorrect representation argument in config")
    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get L2 norm upper bound
    upper_bound = (
        np.max(tf.reduce_sum(x_train**2, axis=list(range(1, x_train.ndim))) ** 0.5)
        * cfg.input_clipping
    )
    # Clip training and testing set imgs
    x_train = tf.clip_by_norm(x_train, upper_bound, axes=list(range(1, x_train.ndim)))
    x_test = tf.clip_by_norm(x_test, upper_bound, axes=list(range(1, x_test.ndim)))
    return x_train, x_test, y_train, y_test, upper_bound


def load_data_mnist(cfg):
    """
    Loads the MNIST dataset and outputs tuple containing x_train,x_test,y_train,y_test,upper_bound. The upper_bound is computed
    the clipped dataset.

    Args :
        cfg : Config object containing the chosen representation allow with the input_clipping factor.

    Returns :
        x_train (training dataset)
        y_train (training labels)
        x_test (testing dataset)
        y_test (testing labels)
        upper_bound (float) : Value of the maximum norm of clipped training dataset.
    """
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = mnist.load_data()
    # Normalize
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get L2 norm upper bound
    upper_bound = (
        np.max(tf.reduce_sum(x_train**2, axis=list(range(1, x_train.ndim))) ** 0.5)
        * cfg.input_clipping
    )
    # Clip training and testing set imgs
    x_train = tf.clip_by_norm(x_train, upper_bound, axes=list(range(1, x_train.ndim)))
    x_test = tf.clip_by_norm(x_test, upper_bound, axes=list(range(1, x_train.ndim)))
    return x_train, x_test, y_train, y_test, upper_bound


def load_data_fashion_mnist(cfg):
    """
    Loads the FashionMNIST dataset and outputs tuple containing x_train,x_test,y_train,y_test,upper_bound. The upper_bound is computed
    the clipped dataset.

    Args :
        cfg : Config object containing the chosen representation allow with the input_clipping factor.

    Returns :
        x_train (training dataset)
        y_train (training labels)
        x_test (testing dataset)
        y_test (testing labels)
        upper_bound (float) : Value of the maximum norm of clipped training dataset.
    """
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = fashion_mnist.load_data()
    # Normalize
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get L2 norm upper bound
    upper_bound = (
        np.max(tf.reduce_sum(x_train**2, axis=list(range(1, x_train.ndim))) ** 0.5)
        * cfg.input_clipping
    )
    # Clip training and testing set imgs
    x_train = tf.clip_by_norm(x_train, upper_bound, axes=list(range(1, x_train.ndim)))
    x_test = tf.clip_by_norm(x_test, upper_bound, axes=list(range(1, x_train.ndim)))
    return x_train, x_test, y_train, y_test, upper_bound
