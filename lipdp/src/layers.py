import tensorflow as tf
import deel
import numpy as np
from abc import abstractmethod

class DPLayer:
   """TODO"""

   @abstractmethod
   def get_DP_LipCoef(self, *args, **kwargs):
      pass

class DP_ResidualSpectralDense(deel.lip.layers.SpectralDense, DPLayer):
    def __init__(self, *args, **kwargs):
      if 'use_bias' in kwargs and kwargs['use_bias']:
        raise ValueError("No bias allowed.")
      kwargs['use_bias']=False
      super().__init__(*args, **kwargs)
    def get_DP_LipCoef(self, *args, **kwargs):
      return 0.5

class DP_SpectralDense(deel.lip.layers.SpectralDense, DPLayer):
    def __init__(self, *args, **kwargs):
      if 'use_bias' in kwargs and kwargs['use_bias']:
        raise ValueError("No bias allowed.")
      kwargs['use_bias']=False
      super().__init__(*args, **kwargs)
    def get_DP_LipCoef(self, *args, **kwargs):
      return 1

class DP_SpectralConv2D(deel.lip.layers.SpectralDense, DPLayer):
    def __init__(self, *args, **kwargs):
      if 'use_bias' in kwargs and kwargs['use_bias']:
        raise ValueError("No bias allowed.")
      kwargs['use_bias']=False
      super().__init__(*args, **kwargs)
    def get_DP_LipCoef(self,*args, **kwargs):
       return self._get_coef() * np.sqrt(np.prod(self.kernel_size))
       


