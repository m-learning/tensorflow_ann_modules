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

def init_network():
  """Initializes network model
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
  
  model = init_network()
  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  
  return model
