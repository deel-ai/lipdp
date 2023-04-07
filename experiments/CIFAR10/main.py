import os 
from ml_collections import config_dict
from ml_collections import config_flags
import deel
import tensorflow as tf
import tensorflow_privacy
import numpy as np
import pandas as pd
import wandb
from absl import app
from absl import flags
from wandb.keras import WandbMetricsLogger,WandbCallback
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from deel.lip.losses import MulticlassHKR, MulticlassKR, MulticlassHinge, TauCategoricalCrossentropy
from deel.lip.layers import SpectralDense,SpectralConv2D,ScaledL2NormPooling2D,ScaledAveragePooling2D
from deel.lip.activations import GroupSort, FullSort
from tensorflow.keras.layers import Input, Flatten
from lipdp.src.losses import KCosineSimilarity,get_lip_constant_loss
from lipdp.src.pipeline import load_data_cifar
from lipdp.src.model import DP_LipNet
from wandb_sweeps.src_config.sweep_config import get_sweep_config


cfg = config_dict.ConfigDict()

cfg.alpha = 50.
cfg.beta_1 = 0.9
cfg.beta_2 = 0.999
cfg.batch_size = 4096
cfg.condense = True
cfg.delta = 1e-5
cfg.epsilon = 500.
cfg.input_clipping = 1.
cfg.K = 1.
cfg.learning_rate = 0.25
cfg.lip_coef = 1.
cfg.loss = "TauCategoricalCrossentropy"
cfg.log_wandb = "disabled"
cfg.min_margin = 0.5
cfg.min_norm = 5.21 
cfg.model_name = "No_name"
cfg.noise_multiplier=0.
cfg.optimizer = "Adam"
cfg.N = 50 * 1000
cfg.num_classes = 10
cfg.opt_iterations = 10
cfg.save = False
cfg.save_folder = os.getcwd()
cfg.steps = 300
cfg.sweep_id = ""
cfg.tau = 8.
cfg.tag = "Default"

_CONFIG = config_flags.DEFINE_config_dict('cfg', cfg)

def create_model(cfg, InputUpperBound):
  # Using VGG architecture of Papernot et Al.
  model = DP_LipNet(
      [
      SpectralConv2D(filters=32, kernel_size=3, input_shape=( 32, 32, 3), kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False), 
      SpectralConv2D(filters=32, kernel_size=3, kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False), 
      ScaledL2NormPooling2D(pool_size=2,strides=2), 
      SpectralConv2D(filters=64, kernel_size=3, kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False),
      SpectralConv2D(filters=64, kernel_size=3, kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False), 
      ScaledL2NormPooling2D(pool_size=2,strides=2), 
      SpectralConv2D(filters=128, kernel_size=3, kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False),
      SpectralConv2D(filters=128, kernel_size=3, kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False),
      ScaledL2NormPooling2D(pool_size=2,strides=2), 
      SpectralConv2D(filters=256, kernel_size=3, kernel_initializer="orthogonal", activation=GroupSort(2), strides=1, use_bias=False),
      ScaledL2NormPooling2D(pool_size=4,strides=4), 
      Flatten(),
      SpectralDense(4096,use_bias=False),
      SpectralDense(10,use_bias=False),
    ],
      k_coef_lip=1.0,
      name="hkr_model",
      X = InputUpperBound,
      cfg = cfg
  )
  return model

def compile_model(model,cfg):
  # Choice of optimizer
  if cfg.optimizer == "SGD" : 
    optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate)
  elif cfg.optimizer == "Adam" : 
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, beta_1=cfg.beta_1, beta_2=cfg.beta_2,epsilon=1e-12)
  else:
    print("Illegal optimizer argument : ", cfg.optimizer)
  # Choice of loss function 
  if cfg.loss == "MulticlassHKR" :
    if cfg.optimizer == "SGD":
      cfg.learning_rate = cfg.learning_rate / cfg.alpha
    loss = MulticlassHKR(alpha=cfg.alpha, min_margin=cfg.min_margin,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  elif cfg.loss == "MulticlassHinge":
    loss = MulticlassHinge(min_margin=cfg.min_margin, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  elif cfg.loss == "MulticlassKR":
    loss = MulticlassKR(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) 
  elif cfg.loss == "TauCategoricalCrossentropy":
    loss = TauCategoricalCrossentropy(cfg.tau,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  elif cfg.loss == "KCosineSimilarity":
    KX_min = cfg.K * cfg.min_norm
    loss = KCosineSimilarity(KX_min, reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  else :
    raise ValueError(f"Illegal loss argument {cfg.loss}")
  # Compile model
  model.compile(
                # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
                # note also in the case of lipschitz networks, more robustness require more parameters.
                loss=loss,
                optimizer=optimizer,
                metrics=["accuracy"],
        ) 
  return model

def train():

  if cfg.log_wandb == 'run':
    wandb.init(project="dp-lipschitz", mode="online", config=cfg)
    
  elif cfg.log_wandb == 'disabled':
    wandb.init(project="dp-lipschitz", mode="disabled", config=cfg)
    
    
  elif cfg.log_wandb.startswith('sweep_'):
    wandb.init() 
    for key, value in wandb.config.items():
      cfg[key] = value
  
  num_epochs = cfg.steps // (cfg.N // cfg.batch_size)
  cfg.noise_multiplier = compute_noise(cfg.N,cfg.batch_size,cfg.epsilon,num_epochs,cfg.delta,1e-6)
  
  x_train,x_test,y_train,y_test,InputUpperBound = load_data_cifar(cfg.input_clipping)
  model = create_model (cfg, InputUpperBound)
  model = compile_model(model,cfg)
  num_epochs = cfg.steps // (cfg.N // cfg.batch_size)
  callbacks = [WandbCallback(monitor='val_accuracy'),EarlyStopping(monitor="val_accuracy",min_delta=0.001,patience=15),
                ReduceLROnPlateau(monitor="val_accuracy",factor=0.9,min_delta=0.0001,patience=5),]
  hist = model.fit(x_train,y_train,epochs=num_epochs,validation_data=(x_test,y_test),batch_size=cfg.batch_size,
                   callbacks=callbacks)
  wandb.log({"Accuracies" : wandb.plot.line_series(
            xs=[np.linspace(0,num_epochs,num_epochs+1),np.linspace(0,num_epochs,num_epochs+1)],
            ys=[hist.history['accuracy'], hist.history['val_accuracy']],
            keys=["Train Accuracy", "Test Accuracy"],
            title="Train/Test Accuracy",
            xname="num_epochs")})
  if cfg.save :
      model.save(f"{cfg.save_folder}/{cfg.model_name}.h5")


def main(_):
  wandb.login()
  if cfg.log_wandb in ['run', 'disabled']:
    train()
  elif cfg.log_wandb.startswith('sweep_'):
    if cfg.sweep_id == "":
      sweep_config = get_sweep_config(cfg)
      sweep_id = wandb.sweep(sweep=sweep_config, project="dp-lipschitz")    
    else:
      sweep_id = cfg.sweep_id    
    wandb.agent(sweep_id, function=train, count=cfg.opt_iterations)

if __name__ == '__main__':
  app.run(main)