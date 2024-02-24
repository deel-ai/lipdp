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
from typing import Callable
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class DatasetMetadata:
    """
    class that handle dataset metadata that will be used
    to compute privacy guarantees
    """

    input_shape: Tuple[int, int, int]
    nb_classes: int
    nb_samples_train: int
    nb_samples_test: int
    class_names: List[str]
    nb_steps_per_epochs: int
    batch_size: int
    max_norm: float


def standardize_CIFAR(image: tf.Tensor):
    """Standardize the image with the CIFAR10 mean and std dev.

    Args:
        image (tf.Tensor): image to standardize of shape (H,W,C) of type tf.float32.
    """
    CIFAR10_MEAN = tf.constant([[[0.4914, 0.4822, 0.4465]]], dtype=tf.float32)
    CIFAR10_STD_DEV = tf.constant([[[0.2023, 0.1994, 0.2010]]], dtype=tf.float32)
    return (image - CIFAR10_MEAN) / CIFAR10_STD_DEV


def get_colorspace_function(colorspace: str):
    if colorspace is None:  # no colorspace transformation
        return lambda x, y: (x, y)
    elif colorspace.upper() == "RGB":
        return lambda x, y: (x, y)
    elif colorspace.upper() == "RGB_STANDARDIZED":
        return lambda x, y: (standardize_CIFAR(x), y)
    elif colorspace.upper() == "HSV":
        return lambda x, y: (tf.image.rgb_to_hsv(x), y)
    elif colorspace.upper() == "YIQ":
        return lambda x, y: (tf.image.rgb_to_yiq(x), y)
    elif colorspace.upper() == "YUV":
        return lambda x, y: (tf.image.rgb_to_yuv(x), y)
    elif colorspace.upper() == "GRAYSCALE":
        return lambda x, y: (tf.image.rgb_to_grayscale(x), y)
    else:
        raise ValueError("Incorrect representation argument in config")


def bound_clip_value(value: float):
    def bound(x, y):
        """clip samplewise"""
        return tf.clip_by_norm(x, value), y

    return bound, value


def bound_normalize() -> Tuple[Callable, float]:
    def bound(x: tf.Tensor, y: tf.Tensor):
        """normalize samplewise"""
        return tf.linalg.l2_normalize(x), y

    return bound, 1.0


@dataclass
class AugmultConfig:
    """Preprocessing options for images at training time.

    Copied from https://github.com/google-deepmind/jax_privacy that was released
    under license Apache-2.0.

    Attributes:
      augmult: Number of augmentation multiplicities to use. `augmult=0`
        corresponds to no augmentation at all, `augmult=1` to standard data
        augmentation (one augmented view per mini-batch) and `augmult>1` to
        having several augmented view of each sample within the mini-batch.
      random_crop: Whether to use random crops for data augmentation.
      random_flip: Whether to use random horizontal flips for data augmentation.
      random_color: Whether to use random color jittering for data augmentation.
      pad: Optional padding before the image is cropped for data augmentation.
    """

    augmult: int
    random_crop: bool
    random_flip: bool
    random_color: bool
    pad: Union[int, None] = 4

    def apply(
        self,
        image: tf.Tensor,
        label: tf.Tensor,
        *,
        crop_size: Sequence[int],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return apply_augmult(
            image,
            label,
            augmult=self.augmult,
            random_flip=self.random_flip,
            random_crop=self.random_crop,
            random_color=self.random_color,
            pad=self.pad,
            crop_size=crop_size,
        )


def padding_input(x: tf.Tensor, pad: int):
    """Pad input image through 'mirroring' on the four edges.

    Args:
      x: image to pad.
      pad: number of padding pixels.
    Returns:
      Padded image.
    """
    x = tf.concat([x[:pad, :, :][::-1], x, x[-pad:, :, :][::-1]], axis=0)
    x = tf.concat([x[:, :pad, :][:, ::-1], x, x[:, -pad:, :][:, ::-1]], axis=1)
    return x


def apply_augmult(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    augmult: int,
    random_flip: bool,
    random_crop: bool,
    random_color: bool,
    crop_size: Sequence[int],
    pad: Union[int, None],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Augmult data augmentation (Hoffer et al., 2019; Fort et al., 2021).

    Copied from https://github.com/google-deepmind/jax_privacy that was released
    under license Apache-2.0.

    Args:
      image: (single) image to augment.
      label: label corresponding to the image (not modified by this function).
      augmult: number of augmentation multiplicities to use. This number
        should be non-negative (this function will fail if it is not).
      random_flip: whether to use random horizontal flips for data augmentation.
      random_crop: whether to use random crops for data augmentation.
      random_color: whether to use random color jittering for data augmentation.
      crop_size: size of the crop for random crops.
      pad: optional padding before the image is cropped.
    Returns:
      images: augmented images with a new prepended dimension of size `augmult`.
      labels: repeated labels with a new prepended dimension of size `augmult`.
    """
    if augmult == 0:
        # No augmentations; return original images and labels with a new dimension.
        images = tf.expand_dims(image, axis=0)
        labels = tf.expand_dims(label, axis=0)
    elif augmult > 0:
        # Perform one or more augmentations.
        raw_image = tf.identity(image)
        augmented_images = []

        for _ in range(augmult):
            image_now = raw_image

            if random_crop:
                if pad:
                    image_now = padding_input(image_now, pad=pad)
                image_now = tf.image.random_crop(image_now, size=crop_size)
            if random_flip:
                image_now = tf.image.random_flip_left_right(image_now)
            if random_color:
                # values copied/adjusted from a color jittering tutorial
                # https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
                image_now = tf.image.random_hue(image_now, 0.1)
                image_now = tf.image.random_saturation(image_now, 0.6, 1.6)
                image_now = tf.image.random_brightness(image_now, 0.15)
                image_now = tf.image.random_contrast(image_now, 0.7, 1.3)

            augmented_images.append(image_now)

        images = tf.stack(augmented_images, axis=0)
        labels = tf.stack([label] * augmult, axis=0)
    else:
        raise ValueError("Augmult should be non-negative.")

    return images, labels


def default_augmult_config(multiplicity: int):
    return AugmultConfig(
        augmult=multiplicity,
        random_flip=True,
        random_crop=True,
        random_color=False,
    )


def load_and_prepare_images_data(
    dataset_name: str = "mnist",
    batch_size: int = 256,
    colorspace: str = "RGB",
    bound_fct: bool = None,
    drop_remainder: bool = True,
    multiplicity: int = 0,
):
    """
    Load dataset_name image dataset using tensorflow datasets.

    Args:
        dataset_name (str): name of the dataset to load.
        batch_size (int): batch size
        colorspace (str): one of RGB, HSV, YIQ, YUV
        drop_remainder (bool, optional): when true drop the last batch if it
            has less than batch_size elements. Defaults to True.
        multiplicity (int): multiplicity of data-augmentation. 0 means no
            augmentation, 1 means standard augmentation, >1 means multiple.
        bound_fct (callable, optional): function that is responsible of
            bounding the inputs. Can be None, bound_normalize or bound_clip_value.
            None means that no clipping is performed, and max theoretical value is
            reported ( sqrt(w*h*c) ). bound_normalize means that each input is
            normalized setting the bound to 1. bound_clip_value will clip norm to
            defined value.

    Returns:
        ds_train, ds_test, metadata: two dataset, with data preparation,
            augmentation, shuffling and batching. Also return an
            DatasetMetadata object with infos about the dataset.
    """
    # load data
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # None bound yield default trivial bound
    nb_classes = ds_info.features["label"].num_classes
    input_shape = ds_info.features["image"].shape
    if bound_fct is None:
        # TODO: consider throwing an error here to avoid unexpected behavior.
        print(
            "No bound function provided, using default bound sqrt(w*h*c) for the input."
        )
        bound_fct = (
            lambda x, y: (x, y),
            float(input_shape[-3] * input_shape[-2] * input_shape[-1]),
        )
    bound_callable, bound_val = bound_fct

    to_float = lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, nb_classes))

    if input_shape[-1] == 1:
        assert (
            colorspace == "grayscale"
        ), "grayscale is the only valid colorspace for grayscale images"
        colorspace = None
    color_space_fun = get_colorspace_function(colorspace)

    ############################
    ####### Train pipeline #####
    ############################

    # train pipeline
    ds_train = ds_train.map(  # map to 0,1 and one hot encode
        to_float,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds_train = ds_train.shuffle(  # shuffle
        min(batch_size * 10, max(batch_size, ds_train.cardinality())),
        reshuffle_each_iteration=True,
    )

    if multiplicity >= 1:
        augmult_config = default_augmult_config(multiplicity)
        crop_size = ds_info.features["image"].shape
        ds_train = ds_train.map(
            lambda x, y: augmult_config.apply(x, y, crop_size=crop_size)
        )
        ds_train = ds_train.unbatch()
    else:
        multiplicity = 1

    ds_train = ds_train.map(  # map colorspace
        color_space_fun,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds_train = ds_train.map(
        bound_callable, num_parallel_calls=tf.data.AUTOTUNE
    )  # apply bound
    ds_train = ds_train.batch(
        batch_size * multiplicity, drop_remainder=drop_remainder
    )  # batch
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ############################
    ####### Test pipeline ######
    ############################

    ds_test = (
        ds_test.map(
            to_float,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            color_space_fun,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(bound_callable, num_parallel_calls=tf.data.AUTOTUNE)  # apply bound
        .shuffle(
            min(batch_size * 10, max(batch_size, ds_test.cardinality())),
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    # get dataset metadata
    metadata = DatasetMetadata(
        input_shape=ds_info.features["image"].shape,
        nb_classes=ds_info.features["label"].num_classes,
        nb_samples_train=ds_info.splits["train"].num_examples,
        nb_samples_test=ds_info.splits["test"].num_examples,
        class_names=ds_info.features["label"].names,
        nb_steps_per_epochs=ds_train.cardinality().numpy()
        if ds_train.cardinality() > 0  # handle case cardinality return -1 (unknown)
        else ds_info.splits["train"].num_examples / batch_size,
        batch_size=batch_size,
        max_norm=bound_val,
    )

    return ds_train, ds_test, metadata


def default_delta_value(dataset_metadata) -> float:
    """Default policy to set delta value.

    Args:
        dataset_metadata (DatasetMetadata): metadata of the dataset.

    Returns:
        float: default delta value.
    """
    n = dataset_metadata.nb_samples_train
    smallest_power10_bigger = 10 ** np.ceil(np.log10(n))
    delta = float(1 / smallest_power10_bigger)
    print(f"Default delta value: {delta}")
    return delta


def download_adbench_datasets(dataset_dir: str):
    import os
    import fsspec

    fs = fsspec.filesystem("github", org="Minqi824", repo="ADBench")
    print(f"Downloading datasets from the remote github repo...")

    save_path = os.path.join(dataset_dir, "datasets", "Classical")
    print(f"Current saving path: {save_path}")

    os.makedirs(save_path, exist_ok=True)
    fs.get(fs.ls("adbench/datasets/" + "Classical"), save_path, recursive=True)


def load_adbench_data(
    dataset_name: str,
    dataset_dir: str,
    standardize: bool = True,
    redownload: bool = False,
):
    """Load a dataset from the adbench package."""
    if redownload:
        download_adbench_datasets(dataset_dir)

    data = np.load(
        f"{dataset_dir}/datasets/Classical/{dataset_name}.npz", allow_pickle=True
    )
    x_data, y_data = data["X"], data["y"]

    if standardize:
        x_data = (x_data - x_data.mean()) / x_data.std()

    return x_data, y_data


def prepare_tabular_data(
    x_train: np.array,
    x_test: np.array,
    y_train: np.array,
    y_test: np.array,
    batch_size: int,
    bound_fct: Callable = None,
    drop_remainder: bool = True,
):
    """Convert Numpy dataset into tensorflow datasets.

    Args:
        x_train (np.array): input data, of shape (N, F) with floats.
        x_test (np.array): input data, of shape (N, F) with floats.
        y_train (np.array): labels in one hot encoding, of shape (N, C) with floats.
        y_test (np.array): labels in one hot encoding, of shape (N, C) with floats.
        batch_size (int): logical batch size
        bound_fct (callable, optional): function that is responsible of
            bounding the inputs. Can be None, bound_normalize or bound_clip_value.
            None means that no clipping is performed, and max theoretical value is
            reported ( sqrt(w*h*c) ). bound_normalize means that each input is
            normalized setting the bound to 1. bound_clip_value will clip norm to
            defined value.
        drop_remainder (bool, optional): when true drop the last batch if it
            has less than batch_size elements. Defaults to True.


    Returns:
        ds_train, ds_test, metadata: two dataset, with data preparation,
            augmentation, shuffling and batching. Also return an
            DatasetMetadata object with infos about the dataset.
    """
    # None bound yield default trivial bound
    nb_classes = np.unique(y_train).shape[0]
    input_shape = x_train.shape[1:]
    bound_callable, bound_val = bound_fct

    ############################
    ####### Train pipeline #####
    ############################

    to_float = lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32))

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(  # shuffle
        min(batch_size * 10, max(batch_size, ds_train.cardinality())),
        reshuffle_each_iteration=True,
    )

    ds_train = ds_train.map(
        bound_callable, num_parallel_calls=tf.data.AUTOTUNE
    )  # apply bound
    ds_train = ds_train.batch(batch_size, drop_remainder=drop_remainder)  # batch
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ############################
    ####### Test pipeline ######
    ############################

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = (
        ds_test.map(bound_callable, num_parallel_calls=tf.data.AUTOTUNE)  # apply bound
        .shuffle(
            min(batch_size * 10, max(batch_size, ds_test.cardinality())),
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    # get dataset metadata
    metadata = DatasetMetadata(
        input_shape=input_shape,
        nb_classes=nb_classes,
        nb_samples_train=x_train.shape[0],
        nb_samples_test=x_test.shape[0],
        class_names=[str(i) for i in range(nb_classes)],
        nb_steps_per_epochs=ds_train.cardinality().numpy()
        if ds_train.cardinality() > 0  # handle case cardinality return -1 (unknown)
        else x_train.shape[0] / batch_size,
        batch_size=batch_size,
        max_norm=bound_val,
    )

    return ds_train, ds_test, metadata
