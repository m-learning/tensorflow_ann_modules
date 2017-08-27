"""
Created on Aug 27, 2017

Trains network model on fashion MNIST data

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import keras

from cnn.famnist import dataset_config as datasets
from cnn.famnist import network_config as networks
from cnn.famnist import training_flags as config
import numpy as np


def init_dataset(flags):
  """Initializes data-set
    Args:
      flags - training configuration flags
    Returns:
      input_shape - input tensor shape
      x_train - training set
      y_train training labels
      x_test - testing set
      y_test - testing labels
  """

  (x_train, y_train, x_test, y_test) = datasets.load_data()
  
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  x_test = np.array(x_test)
  y_test = np.array(y_test)
  
  (img_rows, img_cols) = (flags.image_width, flags.image_height)
  
  if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, flags.img_rows, img_cols)
  else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)
  
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  
  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, flags.num_classes)
  y_test = keras.utils.to_categorical(y_test, flags.num_classes)
  
  return (input_shape, x_train, y_train, x_test, y_test)

def train_network():
  """Trains network model"""
  
  flags = config.read_training_parameters()
  (input_shape, x_train, y_train, x_test, y_test) = init_dataset(flags)
  model = networks.init_and_compile(input_shape, flags.num_classes)
  model.fit(x_train, y_train,
          batch_size=flags.batch_size,
          epochs=flags.epochs,
          verbose=1,
          validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  model.save_weights(flags.weights)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  

if __name__ == '__main__':
  """Initialize and train network model"""
  train_network()
  