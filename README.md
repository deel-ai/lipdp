<!-- Badge section -->
<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.7, 3.8, 3.9-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<!-- Short description of your library -->
<p align="center">
  <b>LipDP</b> is a Python toolkit dedicated to make people happy and fun.


State-of-the-art approaches for training Differentially Private (DP) Deep Neural Networks (DNN) face difficulties to estimate tight bounds on the sensitivity of the network's layers, and instead rely on a process of per-sample gradient clipping. This clipping process not only biases the direction of gradients but also proves costly both in memory consumption and in computation. To provide sensitivity bounds and bypass the drawbacks of the clipping process, we propose to rely on Lipschitz constrained networks. Our theoretical analysis reveals an unexplored link between the Lipschitz constant with respect to their input and the one with respect to their parameters. By bounding the Lipschitz constant of each layer with respect to its parameters, we prove that we can train these networks with privacy guarantees.  Our analysis not only allows the computation of the aforementioned sensitivities at scale, but also provides guidance on how to maximize the gradient-to-noise ratio for fixed privacy guarantees. To facilitate the application of Lipschitz networks and foster robust and certifiable learning under privacy guarantees, we provide this Python package that implements building blocks allowing the construction and private training of such networks.

![backpropforbounds](./docs/assets/backprop_v2.png)

The sensitivity is computed automatically by the package, and no element-wise clipping is required. This is translated into a new DP-SGD algorithm, called Clipless DP-SGD, that is faster and more memory efficient than DP-SGD with clipping.

![speed](./docs/assets/all_speed_curves.png)

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creator](#-creator)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🔥 Tutorials

We propose some tutorials to get familiar with the library and its api:

- [Demo on MNIST](https://colab.research.google.com/github/deel-ai/lipdp/blob/main/docs/notebooks/basic_mnist.ipynb) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/lipdp/blob/main/docs/notebooks/basic_mnist.ipynb) </sub>
- [Demo on CIFAR10](https://colab.research.google.com/github/deel-ai/lipdp/blob/main/docs/notebooks/basic_mnist.ipynb) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/lipdp/blob/main/docs/notebooks/basic_mnist.ipynb) </sub>

## 🚀 Setup

lipDP requires some stuff and several libraries including Numpy. Installation can be
 done using Pypi:

```python
pip install lipdp
```

or locally by cloning the repository and running:
```python
pip install -e .[dev]
```

### Setup privacy parameters

Parameters are stored in a dataclass:

```python
from deel.lipdp.model import DPParameters
dp_parameters = DPParameters(
    noisify_strategy="local",
    noise_multiplier=4.0,
    delta=1e-5,
)

epsilon_max = 10.0
```

### Setup DP model

```python
# construct DP_Sequential
model = DP_Sequential(
    # works like usual sequential but requires DP layers
    layers=[
        # BoundedInput works like Input, but performs input clipping to guarantee input bound
        layers.DP_BoundedInput(
            input_shape=dataset_metadata.input_shape, upper_bound=input_upper_bound
        ),
        layers.DP_QuickSpectralConv2D( # Reshaped Kernel Orthogonalization (RKO) convolution.
            filters=32,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,  # No biases since the framework handles a single tf.Variable per layer.
        ),
        layers.DP_GroupSort(2),  # GNP activation function.
        layers.DP_ScaledL2NormPooling2D(pool_size=2, strides=2),  # GNP pooling.
        layers.DP_QuickSpectralConv2D( # Reshaped Kernel Orthogonalization (RKO) convolution.
            filters=64,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,  # No biases since the framework handles a single tf.Variable per layer.
        ),
        layers.DP_GroupSort(2),  # GNP activation function.
        layers.DP_ScaledL2NormPooling2D(pool_size=2, strides=2),  # GNP pooling.
        
        layers.DP_Flatten(),   # Convert features maps to flat vector.
        
        layers.DP_QuickSpectralDense(512),  # GNP layer with orthogonal weight matrix.
        layers.DP_GroupSort(2),
        layers.DP_QuickSpectralDense(dataset_metadata.nb_classes),
    ],
    dp_parameters=dp_parameters,
    dataset_metadata=dataset_metadata,
)
```

### Setup accountant

The privacy accountant is composed of different mechanisms from `autodp` package that are combined to provide a privacy accountant for Clipless DP-SGD algorithm:

![rdpaccountant](./docs/assets/fig_accountant.png)


Adding a privacy accountant to your model is straighforward:

```python
from deel.lipdp.model import DP_Accountant

callbacks = [
  DP_Accountant()
]
```

## 📦 What's Included

Code can be found in the `lipdp` folder, the documentation ca be found by running
 `mkdocs build` and `mkdocs serve` (or loading `site/index.html`). Experiments were
  done using the code in the `experiments` folder.

Other tools to perform DP-training include:

- [tensorflow-privacy](https://github.com/tensorflow/privacy) in Tensorflow
- [Opacus](https://opacus.ai/) in Pytorch
- [jax-privacy](https://github.com/google-deepmind/jax_privacy) in Jax

## 🙏 Acknowledgments


## 👨‍🎓 Creators


## 🗞️ Citation


## 📝 License

The package is released under [MIT license](LICENSE).
