import numpy as np
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

def get_sweep_config(cfg):
  # Define pertinent parameters according to config : 
  
  header_to_all_sweeps = { 'method': 'bayes',
          'name': 'default',
          'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
          'early_terminate': {'type': 'hyperband', 'min_iter': 10, 'eta': 2}
  }

  common_hyper_parameters = {   'input_clipping': {'max': 1.,'min': 0.2,'distribution': 'log_uniform_values'}, 
                                'batch_size': {'values':[2000,6000,25000],'distribution': 'categorical'},
                                'steps': {'values':[300,500,800],'distribution': 'categorical'},  
  }

  learning_rate_SGD = {'learning_rate': {'max': 0.1,'min': 0.001,'distribution': 'log_uniform_values'},}

  learning_rate_Adam = {'learning_rate': {'max': 0.01,'min': 0.0001,'distribution': 'log_uniform_values'},}

  if cfg.loss == "MulticlassHinge":
    parameters_loss = {'min_margin': {'max': 1.,'min': 0.001,'distribution': 'log_uniform_values'},}

  elif cfg.loss == "MulticlassHKR":
    parameters_loss = {'alpha': {'max': 2000.,'min': 0.01,'distribution': 'log_uniform_values'},
                       'min_margin': {'max': 1.,'min': 0.001,'distribution': 'log_uniform_values'},}

  elif cfg.loss == "MulticlassKR":
    parameters_loss = {}

  elif cfg.loss == "MAE":
    parameters_loss = {}
  
  elif cfg.loss == "TauCategoricalCrossentropy":
    parameters_loss = {'tau': {'max': 18.,'min': 6.,'distribution': 'log_uniform_values'},}

  elif cfg.loss == "KCosineSimilarity":
    parameters_loss = {'K': {'max': 1.,'min': 0.3,'distribution': 'log_uniform_values'},}

  else :
    print('Unrecognised loss functions')

  learning_rate_parameters = learning_rate_SGD if cfg.optimizer == "SGD" else learning_rate_Adam

  assert common_hyper_parameters.keys().isdisjoint(parameters_loss)
  assert common_hyper_parameters.keys().isdisjoint(learning_rate_parameters)
  assert parameters_loss.keys().isdisjoint(learning_rate_parameters)

  sweep_config = {
      **header_to_all_sweeps,
      'parameters': {
          **common_hyper_parameters,
          **parameters_loss,
          **learning_rate_parameters,
      }
  }
    
  epochs = cfg.steps // (cfg.N // cfg.batch_size)
  cfg.noise_multiplier = compute_noise(cfg.N,cfg.batch_size,cfg.epsilon,epochs,cfg.delta,1e-6)
  # Handle sweep
  sweep_name = cfg.log_wandb[len('sweep_'):]
  sweep_config['name'] = sweep_name
  for key, value in cfg.items():
    if key not in sweep_config["parameters"]:
      if key == "loss" :
        print("Loss : ", value)
      sweep_config["parameters"][key] = {'value': value, 'distribution':'constant'}
  # Return the config of sweep :
  return sweep_config