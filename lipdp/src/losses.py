import tensorflow as tf
from keras.losses import LossFunctionWrapper
import numpy as np

@tf.function
def k_cosine_similarity(y_true, y_pred, KX, axis=-1):
  # Cast all values to similar type :
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true = tf.cast(y_true,dtype=tf.float32)
  KX = tf.cast(KX,dtype=tf.float32) 
  # Get minimum value between theoretical and practical value. (TO CHECK)
  factor = tf.where(tf.norm(y_pred) < KX, KX, tf.norm(y_pred))
  y_pred = y_pred / factor
  # Compute and return cosine similarity.
  return -tf.reduce_sum(y_true * y_pred, axis=axis)

class KCosineSimilarity(LossFunctionWrapper):
    def __init__(
        self,
        KX,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="cosine_similarity",
    ):
        super().__init__(
            k_cosine_similarity, KX=KX, reduction=reduction, name=name, axis=axis
        )


# FIRST TRY : TO DEBUG
def get_lip_constant_loss (cfg) :
    if cfg.loss in ["MulticlassHinge","MulticlassKR","CategoricalCrossentropy"]:
        L = 1
    elif cfg.loss == "MulticlassHKR":
        L = cfg.alpha + 1
    elif cfg.loss == "TauCategoricalCrossentropy":
        L = cfg.tau * np.sqrt(cfg.num_classes-1)/cfg.num_classes
    elif cfg.loss == "KCosineSimilarity" :
        L = 1/float(cfg.K * cfg.min_norm)
    else :
        raise TypeError(f"Unrecognised Loss Function Argument {cfg.loss}")
    print("Lipschitz constant of chosen loss function is : ", L)
    return L