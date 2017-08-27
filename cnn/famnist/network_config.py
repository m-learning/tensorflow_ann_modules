"""
Created on Aug 27, 2017

Network model for fashion MNIST training and interface

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import (Input, Conv2D, MaxPool2D, \
                          Activation, Flatten, Dense, \
                          BatchNormalization, Dropout)
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Nadam


SAME = 'same'
RELU = 'relu'
SOFTMAX = 'softmax'
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
POOL_STRIDE = (2, 2)

def _add_dropout(keep_prob, net, is_training):
  """Adds dropout layer to model
    Args:
      keep_prob - keep probability
      net - network model
      is_training - training flag
    Returns:
      net - network model
  """
  
  if is_training:
    result = Dropout(keep_prob)(net)
  else:
    result = net
  
  return result

def init_model(input_shape, num_classes, is_training=False):
  """Initializes network model
    Args:
      input_shape - input tensor shape
      num_classes - number of output classes
      is_training - flag for training
    Returns:
      model - initialized network model
  """
  
  input_image = Input(shape=input_shape)
  net = Conv2D(32, KERNEL_SIZE, padding=SAME)(input_image)
  net = BatchNormalization()(net, training=is_training)
  net = Activation(RELU)(net)
  net = MaxPool2D(POOL_SIZE, POOL_STRIDE)(net)
  net = Conv2D(64, KERNEL_SIZE, padding=SAME)(net)
  net = BatchNormalization()(net, training=is_training)
  net = Activation(RELU)(net)
  net = MaxPool2D(POOL_SIZE, POOL_STRIDE)(net)
  net = _add_dropout(0.25, net, is_training)
  net = Flatten()(net)
  net = Dense(1024)(net)
  net = BatchNormalization()(net, training=is_training)
  net = Activation(RELU)(net)
  net = _add_dropout(0.5, net, is_training)
  logits = Dense(num_classes)(net)
  prediction = Activation(SOFTMAX)(logits)
  model = Model(inputs=input_image, outputs=prediction)
  
  return model

def init_and_compile(input_shape, num_classes):
  """Initializes and compiles network
    Args:
      input_shape - input tensor shape
      num_classes - number of output classes
    Returns:
      model - initialized network model      
  """
  
  model = init_model(input_shape, num_classes, is_training=True)
  model.compile(loss=categorical_crossentropy,
              optimizer=Nadam(),
              metrics=['accuracy'])
  
  return model
