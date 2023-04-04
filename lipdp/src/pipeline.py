import numpy as np
import tensorflow as tf
import deel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

def load_data_cifar(input_clipping):
    # Load data
    (x_train, y_train_ord), (x_test, y_test_ord) = cifar10.load_data()
    # Normalize
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    # One hot labels for multiclass classifier
    y_train = to_categorical(y_train_ord)
    y_test = to_categorical(y_test_ord)
    # Get theoretical L2 norm upper bound
    theoretical_upper_bound = np.sqrt(np.prod(x_train[0].shape))
    upper_bound = theoretical_upper_bound * input_clipping
    # Clip training and testing set imgs 
    x_train = tf.clip_by_norm(x_train,upper_bound,axes=list(range(1,x_train.ndim)))
    x_test = tf.clip_by_norm(x_test,upper_bound,axes=list(range(1,x_train.ndim)))
    return x_train,x_test,y_train,y_test,upper_bound

def load_data_mnist(input_clipping):
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
    upper_bound = theoretical_upper_bound * input_clipping
    # Clip training and testing set imgs 
    x_train = tf.clip_by_norm(x_train,upper_bound,axes=list(range(1,x_train.ndim)))
    x_test = tf.clip_by_norm(x_test,upper_bound,axes=list(range(1,x_train.ndim)))
    return x_train,x_test,y_train,y_test,upper_bound

def load_data_fashion_mnist(input_clipping):
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
    upper_bound = theoretical_upper_bound * input_clipping
    # Clip training and testing set imgs 
    x_train = tf.clip_by_norm(x_train,upper_bound,axes=list(range(1,x_train.ndim)))
    x_test = tf.clip_by_norm(x_test,upper_bound,axes=list(range(1,x_train.ndim)))
    return x_train,x_test,y_train,y_test,upper_bound
