import deel
import tensorflow as tf
from lipdp.src.losses import get_lip_constant_loss 
import numpy as np

def map_trainable_layers (model):
   """Is a verification function allowing to map layers to their trainable character.
   Args : 
    model: Sequential like model
   Returns :
    list : A list of boolean values of the same length as model.layers. If the boolean value is true
    the layer of corresponding index is a trainable layer.
    """
   layer_names = [layer.name for layer in model.layers]
   var_names = [var.name.split("/")[0] for var in model.trainable_variables]
  #  print('layer_names', layer_names)
  #  print('var_names', var_names)
   map_vars = [l in var_names for l in layer_names]
   return map_vars

def get_layer_coefs(model):
    """
    Computes the Lipschitz constant of known type layers.
    Args : 
      model: Contains the Sequential based, DP_LipNet model class.
    Returns :
      Dictionary containing the values of the Lipschitz constants for each layer name.
    """
    dict_coef = {}
    for layer in model.layers :
        if "spectral_dense" in layer.name:
            if layer.bias is None :
              dict_coef[layer.name] = 1
            else : 
              raise TypeError(f"Illegal use_bias argument for layer {layer.name}")
        elif "spectral_conv2d" in layer.name:
            if layer.bias is None :
              dict_coef[layer.name] = layer._get_coef() * np.sqrt(np.prod(layer.kernel_size))
            else : 
              raise TypeError(f"Illegal use_bias argument for layer {layer.name}")

    return dict_coef
   
def get_noise(cfg, trainable_vars, gradients, X, trainable_layer_coefs):
  """
  Returns the noise tensor necessary to guarantee (epsilon,delta)-DP according to the config : 
  Args:
    cfg: Experiment configuration (contains batch_size & noise_multiplier).
    trainable_vars: Trainable variables of the model.
    gradients: Computed parameter wise gradients.
    X: Maximum input norm.
    trainable_layer_coefs: Lipschitz coefficient associated with each trainable layer.
  Returns:
    Noise tensor with pertinent standard deviation to guarantee (epsilon,delta)-DP for 1-Lipschitz, Gradient Norm preserving model.
  """
  # Get the L constant of the chosen loss :
  L = get_lip_constant_loss(cfg)
  # Generate the noise contaning array
  noise = []
  # Initialise convolution counter
  for grad,var in zip(gradients,trainable_vars) :
    name = var.name.split("/")[0]
    K = L * X * trainable_layer_coefs[name]
    stddev = cfg.noise_multiplier * 2 * K / cfg.batch_size 
    print(f"Adding noise of stddev {stddev} on gradient of {name} of maximum value {K}")
    # Get corresponding noise Tensor :  
    noise.append(tf.random.normal(shape=grad.shape, stddev=stddev))
  return noise

def noisify_gradients(model, trainable_variables, gradients, map_vars) :
  """
  Returns the (epsilon,delta)-DP gradients of the DP_LipNet model during the training step.
  Args : 
    model: DP_LipNet model.
    trainable_variables: Trainable layers of the model.
    gradients: Computed gradients during the training step.
  Returns : 
    Noisy version of the gradients guaranteeing (epsilon,delta)-DP for 1-Lipschitz, Gradient Norm Preserving model.
  """
  trainable_layer_coefs = get_layer_coefs(model)
  # print(len(trainable_variables))
  # print(len(gradients))
  assert(sum(map_vars)==len(trainable_layer_coefs))
  # Compute noise
  assert(len(trainable_variables)==len(gradients))
  noise = get_noise(model.cfg, trainable_variables, gradients, model.X, trainable_layer_coefs)
  assert(len(noise)==len(gradients))
  # Add noise
  noisy_gradients = [g+n for g, n in zip(gradients, noise)]
  # for g,n in zip(gradients,noise):
  #   print(f"Added noise {n} to gradient {g}")
  return noisy_gradients

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
    def __init__(self, *args, X, cfg, **kwargs):
      self.X = X
      self.cfg = cfg
      super().__init__(*args, **kwargs)

    # Define the differentially private training step
    def train_step(self, data):
        # Unpack data
        x, y = data

        map_vars = map_trainable_layers(self) 

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
        
        noisy_gradients = noisify_gradients(self, trainable_vars, gradients, map_vars)
        
        # Each optimizer is a postprocessing of the already (epsilon,delta)-DP gradients
        self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update Metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Condense to verify |W|_2 = 1
        if self.cfg.condense : 
          self.condense()
        return {m.name: m.result() for m in self.metrics}