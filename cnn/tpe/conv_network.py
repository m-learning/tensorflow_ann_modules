"""
Created on Jan 24, 2017

Network model for convolutional face detection

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Flatten, Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential


def _add_dropout(model, is_training, keep_prob=0.5):
  """Adds dropout layer to model if is training
    Args:
      model - network model
      is_training - flag for training
      keep_prob - keep probability parameter
  """
  
  if is_training:
    model.add(Dropout(keep_prob))
  
def init_cnn(dim, n_classes, is_training=False):
  """Builds network model
    Args:
      dim - image dimensions
      n_classes - number of output classes
    Returns:
      model - network model
  """  
  
  model = Sequential()

  model.add(Convolution2D(96, 11, 11,
                          subsample=(4, 4),
                          input_shape=(dim, dim, 3),
                          init='glorot_uniform',
                          border_mode='same'))
  model.add(PReLU())
  model.add(MaxPooling2D((3, 3), strides=(2, 2)))

  model.add(Convolution2D(256, 5, 5,
                          subsample=(1, 1),
                          init='glorot_uniform',
                          border_mode='same'))
  model.add(PReLU())
  model.add(MaxPooling2D((3, 3), strides=(2, 2)))

  model.add(Convolution2D(384, 3, 3,
                          subsample=(1, 1),
                          init='glorot_uniform',
                          border_mode='same'))
  model.add(PReLU())

  model.add(Convolution2D(384, 3, 3,
                          subsample=(1, 1),
                          init='glorot_uniform',
                          border_mode='same'))
  model.add(PReLU())

  model.add(Convolution2D(256, 3, 3,
                          subsample=(1, 1),
                          init='glorot_uniform',
                          border_mode='same'))
  model.add(PReLU())
  model.add(MaxPooling2D((3, 3), strides=(2, 2)))

  model.add(Flatten())
  _add_dropout(model, is_training)

  model.add(Dense(2048, init='glorot_uniform'))
  model.add(PReLU())
  _add_dropout(model, is_training)

  model.add(Dense(256, init='glorot_uniform'))
  model.add(PReLU())

  model.add(Dense(n_classes, init='glorot_uniform', activation='softmax'))
  
  return model
  
def build_cnn(dim, n_classes):
  """Builds network model
    Args:
      dim - image dimensions
      n_classes - number of output classes
    Returns:
      model - network model
  """
  model = init_cnn(dim, n_classes, is_training=True)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model