import deel
import tensorflow as tf
from deel.lipdp.layers import DPLayer
from deel.lipdp.losses import get_lip_constant_loss
import numpy as np

def global_noisify(model, trainable_vars, gradients) :
  sum_bounds = 0
  L = get_lip_constant_loss(model.cfg)
  for layer in model.layers :
     #DEBUG FOR CONVOLUTIONS
     if isinstance(layer, DPLayer) :
        sum_bounds += (layer.get_DP_LipCoef() * L * model.X) **2
        # tf.print(f"{layer.name} has DP coef of {layer.get_DP_LipCoef()}") 
  global_bound = np.sqrt(sum_bounds)
  stddev = model.cfg.noise_multiplier * global_bound / model.cfg.batch_size
  # print(f"Adding noise of stddev : {stddev}")
  noises = [tf.random.normal(shape=g.shape, stddev=stddev) for g in gradients]
  noisy_grads = [g+n for g,n in zip(gradients,noises)]
  return noisy_grads

class DP_LipNet (deel.lip.model.Sequential):
  """"Model Class based on the DEEL Sequential model. Takes into account the architecture, only the following components are allowed :
  - Input
  - SpectralDense
  - SpectralConv2D
  - Flatten
  - ScaledL2GlobalPooling
  Args :
    Architecture: Sequential like input
    cfg: Contains the experiment config, must contain the batch size information and the chosen noise multiplier.
    X: The previously determined maximum norm among the training set.
  The model is then calibrated to verify (epsilon,delta)-DP guarantees by noisying the values of the gradients during the training step. 
  Do not modify the config object after the model has been declared with config object cfg. 
  Hypothesis: The model is 1-Lipschitz and Dense layers are Gradient Norm Preserving.
  """ 
  def __init__(self, *args, X, noisify_strategy, cfg, **kwargs):
    if noisify_strategy == "global":
        self.noisify_fun = global_noisify
    # elif noisify_strategy == "local" :
    #     self.noisify_fun = local_noisify
    else : 
        raise TypeError("Incorrect noisify_strategy argument during model initialisation.")
    self.X = X
    self.cfg = cfg
    super().__init__(*args, **kwargs)

  # Define the differentially private training step
  def train_step(self, data):
      # Unpack data
      x, y = data

      with tf.GradientTape() as tape:
          y_pred = self(x, training=True)  # Forward pass
          # tf.cast(y_pred,dtype=y.dtype)
          # Compute the loss value
          # (the loss function is configured in `compile()`)
          loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

      # Compute gradients
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      # Get lipschitz coefficient of layers from gradients  
      print(f"Noise function is {str(self.noisify_fn)}")
      noisy_gradients = self.noisify_fun(self, trainable_vars, gradients)
      # Each optimizer is a postprocessing of the already (epsilon,delta)-DP gradients
      self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))
      # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      # Update Metrics
      self.compiled_metrics.update_state(y, y_pred)
      # Condense to verify |W|_2 = 1
      if self.cfg.condense : 
        self.condense()
      return {m.name: m.result() for m in self.metrics}

class DP_LipNet_Model (deel.lip.model.Model):
  """"Model Class based on the DEEL Sequential model. Takes into account the architecture, only the following components are allowed :
  - Input
  - SpectralDense
  - SpectralConv2D
  - Flatten
  - ScaledL2GlobalPooling
  Args :
    Architecture: Sequential like input
    cfg: Contains the experiment config, must contain the batch size information and the chosen noise multiplier.
    X: The previously determined maximum norm among the training set.
  The model is then calibrated to verify (epsilon,delta)-DP guarantees by noisying the values of the gradients during the training step. 
  Do not modify the config object after the model has been declared with config object cfg. 
  Hypothesis: The model is 1-Lipschitz and Dense layers are Gradient Norm Preserving.
  """ 
  def __init__(self, *args, X, noisify_strategy, cfg, **kwargs):
    if noisify_strategy == "global":
        self.noisify_fun = global_noisify
    elif noisify_strategy == "local" :
        self.noisify_fun = local_noisify
    else : 
        raise TypeError("Incorrect noisify_strategy argument during model initialisation.")
    
    self.X = X
    self.cfg = cfg
    super().__init__(*args, **kwargs)

  # Define the differentially private training step
  def train_step(self, data):
      # Unpack data
      x, y = data

      with tf.GradientTape() as tape:
          y_pred = self(x, training=True)  # Forward pass
          # tf.cast(y_pred,dtype=y.dtype)
          # Compute the loss value
          # (the loss function is configured in `compile()`)
          loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

      # Compute gradients
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      # Get lipschitz coefficient of layers from gradients     
      noisy_gradients = self.noisify_fun(self, trainable_vars, gradients) 
      # Each optimizer is a postprocessing of the already (epsilon,delta)-DP gradients
      self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))
      # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      # Update Metrics
      self.compiled_metrics.update_state(y, y_pred)
      # Condense to verify |W|_2 = 1
      if self.cfg.condense : 
        self.condense()
      return {m.name: m.result() for m in self.metrics}
