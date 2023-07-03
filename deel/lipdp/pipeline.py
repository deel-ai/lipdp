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
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

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


def get_colorspace_function(colorspace: str):
    if colorspace.upper() == "RGB":
        return lambda x, y: (x, y)
    elif colorspace.upper() == "HSV":
        return lambda x, y: (tf.image.rgb_to_hsv(x), y)
    elif colorspace.upper() == "YIQ":
        return lambda x, y: (tf.image.rgb_to_yiq(x), y)
    elif colorspace.upper() == "YUV":
        return lambda x, y: (tf.image.rgb_to_yuv(x), y)
    else:
        raise ValueError("Incorrect representation argument in config")


def bound_clip_value(value):
    def bound(x, y):
        """clip samplewise"""
        return tf.clip_by_norm(x, value), y

    return bound, value


def bound_normalize():
    def bound(x, y):
        """normalize samplewise"""
        return tf.linalg.l2_normalize(x), y

    return bound, 1.0


def augment(ds, augmentations, batch_size, drop_remainder):
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    list_x = []
    list_y = []
    aug_models = [
        tf.keras.Sequential([aug]) for aug in augmentations if not isinstance(aug, list)
    ]
    multiple_aug = [
        tf.keras.Sequential(aug) for aug in augmentations if isinstance(aug, list)
    ]
    for element in multiple_aug:
        aug_models.append(element)
    for batch in ds:
        x_batch, y_batch = batch
        x_aug = tf.concat([aug_model(x_batch) for aug_model in aug_models], axis=0)
        x_aug = tf.concat([x_batch, x_aug], axis=0)
        y_aug = tf.concat([y_batch for _ in range(len(augmentations) + 1)], axis=0)
        list_x.append(x_aug)
        list_y.append(y_aug)
    aug_ds = tf.data.Dataset.from_tensor_slices((list_x, list_y))
    return aug_ds


def load_and_prepare_data(
    dataset_name: str = "mnist",
    batch_size: int = 256,
    augmentations: Optional[Sequence[tf.keras.layers.Layer]] = None,
    colorspace: str = "RGB",
    random_flip: bool = True,
    random_crop: bool = True,
    crop_size: Optional[Sequence[int]] = None,
    pad: Optional[int] = None,
    drop_remainder=True,
    bound_fct=None,
):
    """
    load dataset_name data using tensorflow datasets.

    Args:
        dataset_name (str): name of the dataset to load.
        batch_size (int): batch size
        augmentations (list of layer objects) : list of augmentations you want to append to
        the batch.
        colorspace (str): one of RGB, HSV, YIQ, YUV
        drop_remainder (bool, optional): when true drop the last batch if it
            has less than batch_size elements. Defaults to True.
        augmentation_fct (callable, optional): data augmentation to be applied
            to train. the function must take a tuple (img, label) and return a
            tuple of (img, label). Defaults to None.
        bound_fct (callable, optional): function that is responsible of
            bounding the inputs. Can be None, bound_normalize or bound_clip_value.
            None means that no clipping is performed, and max theoretical value is
            reported ( sqrt(w*h*c) ). bound_normalize means that each input is
            normalized setting the bound to 1. bound_clip_value will clip norm to
            defined value.

    Returns:
        ds_train, ds_test, metadat: two dataset, with data preparation,
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
        bound_fct = (
            lambda x, y: (x, y),
            input_shape[-3] * input_shape[-2] * input_shape[-1],
        )
    bound_callable, bound_val = bound_fct

    ds_train = (
        ds_train.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, nb_classes)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(  # map colorspace
            get_colorspace_function(colorspace),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(bound_callable, num_parallel_calls=tf.data.AUTOTUNE)  # apply bound
    )

    if augmentations is not None:
        ds_train = augment(ds_train, augmentations, batch_size, drop_remainder)
    else:
        ds_train = ds_train.batch(batch_size, drop_remainder=drop_remainder)

    # train pipeline
    ds_train = ds_train.shuffle(  # shuffle
        min(batch_size * 10, max(batch_size, ds_train.cardinality())),
        reshuffle_each_iteration=True,
    ).prefetch(tf.data.AUTOTUNE)

    ds_test = (
        ds_test.map(
            lambda x, y: (
                tf.cast(x, tf.float32) / 255.0,
                tf.one_hot(y, nb_classes),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            get_colorspace_function(colorspace),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(bound_callable, num_parallel_calls=tf.data.AUTOTUNE)  # apply bound
        .shuffle(
            min(batch_size * 10, max(batch_size, ds_test.cardinality())),
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=drop_remainder)
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
