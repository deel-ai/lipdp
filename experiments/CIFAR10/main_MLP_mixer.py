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
from deel.lip.layers import ScaledL2NormPooling2D,ScaledAveragePooling2D
from lipdp.src.layers import DP_SpectralConv2D,DP_SpectralDense,DP_ResidualSpectralDense
from deel.lip.activations import GroupSort, FullSort
from tensorflow.keras.layers import Input, Flatten
from lipdp.src.losses import KCosineSimilarity,get_lip_constant_loss
from lipdp.src.pipeline import load_data_cifar
from lipdp.src.model import DP_LipNet_Model
from wandb_sweeps.src_config.sweep_config import get_sweep_config


cfg = config_dict.ConfigDict()

cfg.alpha = 50.
cfg.beta_1 = 0.9
cfg.beta_2 = 0.999
cfg.batch_size = 2000
cfg.condense = True
cfg.delta = 1e-5
cfg.epsilon = 7.53
cfg.hidden_size = 700
cfg.input_clipping = 0.5
cfg.K = 0.99
cfg.learning_rate = 1e-3
cfg.lip_coef = 1.
cfg.loss = "MAE"
cfg.log_wandb = "disabled"
cfg.min_margin = 0.5
cfg.min_norm = 5.21 
cfg.mlp_channel_dim = 256
cfg.mlp_seq_dim = 256
cfg.model_name = "No_name"
cfg.noise_multiplier = 0.
cfg.num_mixer_layers = 5
cfg.optimizer = "Adam"
cfg.patch_size = 2
cfg.N = 50 * 1000
cfg.num_classes = 10
cfg.opt_iterations = 10
cfg.run_eagerly = False
cfg.save = False
cfg.save_folder = os.getcwd()
cfg.steps = 2500
cfg.sweep_id = ""
cfg.tau = 3.
cfg.tag = "Default"

_CONFIG = config_flags.DEFINE_config_dict('cfg', cfg)

def mlp_block_lip(x, mlp_dim):
    y = DP_ResidualSpectralDense(mlp_dim, use_bias=False)(x)
    y = GroupSort(2)(y)
    return DP_ResidualSpectralDense(x.shape[-1],use_bias=False)(y)

def mixer_block_lip(x, tokens_mlp_dim, channels_mlp_dim):
    y = tf.keras.layers.Permute((2, 1))(x)
    
    token_mixing = mlp_block_lip(y, tokens_mlp_dim)
    token_mixing = tf.keras.layers.Permute((2, 1))(token_mixing)
    x = 0.5 * tf.keras.layers.Add()([x, token_mixing])
    
    channel_mixing = mlp_block_lip(y, channels_mlp_dim)
    #ADDED
    channel_mixing = tf.keras.layers.Permute((2, 1))(channel_mixing)
    output = tf.keras.layers.Add()([x, channel_mixing])
    return output

def mlp_mixer_lip(x, loss, num_blocks, patch_size, hidden_dim, 
              tokens_mlp_dim, channels_mlp_dim,
              num_classes=10):
    x = tf.image.extract_patches(images = x, sizes=[1,patch_size,patch_size,1], strides=[1,patch_size,patch_size,1], rates=[1, 1, 1, 1], padding="VALID")
    # x = DP_SpectralConv2D(filters=hidden_dim, kernel_size=patch_size, use_bias=False, strides=patch_size, padding="same")(x)
    x = tf.keras.layers.Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)

    for _ in range(num_blocks):
        x = mixer_block_lip(x, tokens_mlp_dim, channels_mlp_dim)
    
    # TO REPLACE ?
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return DP_SpectralDense(num_classes, use_bias=False, dtype="float32")(x)

def create_model(cfg,InputUpperBound):
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    outputs = mlp_mixer_lip(inputs, cfg.loss, cfg.num_mixer_layers,cfg.patch_size, cfg.hidden_size, cfg.mlp_seq_dim, cfg.mlp_channel_dim)
    return DP_LipNet_Model(inputs, outputs, X=InputUpperBound, cfg=cfg, noisify_strategy="global", name="mlp_mixer")

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
  elif cfg.loss == "MAE":
    loss = tf.keras.losses.MeanAbsoluteError(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  else :
    raise ValueError(f"Illegal loss argument {cfg.loss}")
  # Compile model
  model.compile(
                # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
                # note also in the case of lipschitz networks, more robustness require more parameters.
                loss=loss,
                optimizer=optimizer,
                metrics=["accuracy"],
                run_eagerly=cfg.run_eagerly,
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
  model.summary()
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