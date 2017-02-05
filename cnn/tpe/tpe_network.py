"""
Created on Jan 24, 2017

Network model for TPE distances

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Lambda, Input, merge
from keras.models import Model, Sequential

import keras.backend as K
import numpy as np


# from keras.optimizers import SGD
def triplet_loss(y_true, y_pred):
  """Calculates triplet loss
    Args:
      y_true - true label
      y_pred - predicted value
    Returns:
      loss value
  """
  return -K.mean(K.log(K.sigmoid(y_pred)))

def triplet_merge(inputs):
  """Merges triplet
    Args:
      inputs - input triplet
    Returns:
      merged_result - merged value
  """
  (a, p, n) = inputs
  merged_result = K.sum(a * (p - n), axis=1)
  
  return merged_result


def triplet_merge_shape(input_shapes):
  """Gets merged triplet shape
    Args:
      input_shapes - triplet shapes
    Returns:
      shape of merged triplet
  """
  return (input_shapes[0][0], 1)

def build_tpe(n_in, n_out, W_pca=None):
  """Builds TPE model
    Args:
      n_in - number of inputs
      n_out - number of outputs
      w_pca - network weights
    Returns:
      tuple of -
        model - network model
        predict - prediction function
  """
    
  a = Input(shape=(n_in,))
  p = Input(shape=(n_in,))
  n = Input(shape=(n_in,))

  if W_pca is None:
    W_pca = np.zeros((n_in, n_out))

  base_model = Sequential()
  base_model.add(Dense(n_out, input_dim=n_in, bias=False, weights=[W_pca], activation='linear'))
  base_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

  a_emb = base_model(a)
  p_emb = base_model(p)
  n_emb = base_model(n)

  e = merge([a_emb, p_emb, n_emb], mode=triplet_merge, output_shape=triplet_merge_shape)

  model = Model(input=[a, p, n], output=e)
  predict = Model(input=a, output=a_emb)

  model.compile(loss=triplet_loss, optimizer='rmsprop')

  return (model, predict)
