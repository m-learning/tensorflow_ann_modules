"""
Created on Feb 5, 2017

Utility module for vector compare

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def compare_many(self, dist, xs, ys):
  """Compares two face embedding vectors
    Args:
      dist = distance threshold
      xs - face embedding vectors
      ys - face embedding vectors
    Returns:
      result - vector compare result
  """
    
  xs = np.array(xs)
  ys = np.array(ys)
  scores = np.linalg.norm(xs, ys.T)
  result = (scores, scores > dist)
  
  return result
