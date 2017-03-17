"""
Created on Mar 17, 2017

Data logger for training and evaluation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def log_model(flags, model):
  """Summarizes model if verbose is set to true
    Args:
      flags - model configuration parameters
      model - network model
  """
  
  if flags.verbose:
    print(model.summary())
