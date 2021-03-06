"""
Created on Jan 21, 2017

Classifier for MNIST on Keras library

Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils

from cnn.mnist.cnn_files import training_file
import numpy as np


np.random.seed(1337)  # for reproducibility


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of filters to use
first_conv_filters = 32
second_conv_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# kernel size
kernel_size = (3, 3)

WEIGHTS_FILE = 'keras_weights.h5'

_files = training_file()
weights_path = _files.join_path(_files.model_dir, WEIGHTS_FILE)
# the data, shuffled and split between train and test sets
((x_train, y_train), (x_test, y_test)) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
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
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

class mnist_model(object):
  """Defines MNIST network model"""
  
  def __init__(self, is_training=False):
    self._is_training = is_training
    self.model = None
    
  def _add_dropout(self, prob=0.5):
    """Adds dropout layer to model
      Args:
        prob - keep probability
    """
    if self._is_training:
      self.model.add(Dropout(prob))
  
  def _init_model(self):
    """Defines MNIST network model"""
    self.model = Sequential()

    self.model.add(Conv2D(first_conv_filters, kernel_size=kernel_size,
                          activation='relu',
                          input_shape=input_shape))
    self.model.add(Conv2D(second_conv_filters, kernel_size=kernel_size, activation='relu'))
    self.model.add(MaxPooling2D(pool_size=pool_size))
    self._add_dropout(prob=0.25)
    
    self.model.add(Flatten())
    self.model.add(Dense(128, activation='relu'))
    self._add_dropout(prob=0.5)
    self.model.add(Dense(num_classes, activation='softmax'))
    
  @property
  def network_model(self):
    """Gets MNIST network model
      Returns:
        network model
    """
    
    if self.model is None:
      self._init_model()
    
    return self.model
  
  def _load_model_and_weights(self):
    """Gets MNIST network model and loads weights
      Returns:
        network model
    """
    if self.model is None:
      self._is_training = False
      self._init_model()
      self.model.load_weights(weights_path)
    
    return self.model    
  
  def run_model(self, x):
    """Runs model interface
      Args:
        x - model input
      Returns:
        pred - model predictions
    """
    _network_model = self._load_model_and_weights()
    pred = _network_model(x)
    
    return pred
    
_model_init = mnist_model(is_training=True)
model = _model_init.network_model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save_weights(weights_path)

