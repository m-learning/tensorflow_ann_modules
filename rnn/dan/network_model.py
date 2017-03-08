"""
Created on Jan 27, 2017

Network model for DIM three layer implementation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling1D
from keras.layers import Embedding
from keras.models import Sequential

from rnn.dan import network_config as config


def _add_dropout(model, is_training=False):
  """Adds dropout layer to model
    Args:
      model - network model
      is_training - training flag for regularization
  """
  if is_training:
      model.add(Dropout(config.dropout))

def dan_model(is_training=False):
  """Initializes DAN network model
    Args:
      is_training - training flag for regularization
    Returns:
      model - implemented model
  """
  model = Sequential()
  
  # we start off with an efficient embedding layer which maps
  # our vocab indices into embedding_dims dimensions
  if is_training:
    model.add(Embedding(config.max_features,
                      config.embedding_dims,
                      input_length=config.maxlen,
                      dropout=config.dropout))
  else:
    model.add(Embedding(config.max_features,
                      config.embedding_dims,
                      input_length=config.maxlen)) 
  
  # Averaging
  model.add(GlobalAveragePooling1D())
  
  # Deep networks
  for _ in range(config.nb_hidden_layers):
    model.add(Dense(config.embedding_dims))
    _add_dropout(model, is_training=is_training)
    model.add(Activation('relu'))
  
  # final output
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  
  return model
