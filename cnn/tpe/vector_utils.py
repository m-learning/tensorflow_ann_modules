"""
Created on Feb 5, 2017

Utility module for vector compare

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def compare_many(dist, xs, ys):
  """Compares two face embedding vectors
    Args:
      dist = distance threshold
      xs - face embedding vectors
      ys - face embedding vectors
    Returns:
      tuple of -
      scores - scores for matrices
      boolean flags
  """
    
  xs = np.array(xs)
  ys = np.array(ys)
  scores = np.matmul(xs, ys.T)
  result = (scores, scores > dist)
  
  return result
