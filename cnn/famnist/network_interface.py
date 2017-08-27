"""
Created on Aug 27, 2017

Network interface for fashion MNIST images

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.famnist import network_config as networks

def run_prediction(flags, inp_image):
  """Runs prediction on image
    Args:
      flags - configuration parameters
      inp_image - input image tensor
    Returns:
      predictions - predictions on image
  """
  
  input_shape = networks.init_input_shape(flags)
  model = networks.init_model(input_shape, flags.num_classes, is_training=False)
  predictions = model(inp_image)
  
  return predictions
