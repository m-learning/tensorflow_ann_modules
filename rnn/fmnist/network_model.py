"""
Created on Feb 10, 2017

Network model for speech recognition

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import LSTM, Dense
from keras.models import Sequential


width = 20  # mfcc features
height = 80  # (max) length of utteranc
classes = 10  # digits

def _add_LSTM_cell(is_training, model):
  """Adds LSTM cell to model
    Args:
      is_training - training flag
      model - network model
  """
  
  if is_training:
    model.add(LSTM(128, input_shape=(width, height), dropout_W=0.8))
  else:
    model.add(LSTM(128, input_shape=(width, height)))

def network_network(is_training=True):
  """Initializes network model
    Args:
      is_training - training flag
    Returns:
      model - network model
  """
  
  model = Sequential()
  model.add(LSTM(128, input_shape=(width, height), dropout_W=0.8))
  model.add(Dense(classes, activation='softmax'))
  
  return model

def pretrain_network():
  """Initializes and prepares network model for training
    Returns:
      model - network model
  """
  
  model = network_network()
  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  
  return model
