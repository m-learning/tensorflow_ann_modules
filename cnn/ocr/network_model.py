"""
Created on Feb 18, 2017

Network model for OCR module

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.models import Model

from cnn.ocr.network_config import img_h, conv_num_filters, filter_size, pool_size, rnn_size, time_dense_size, act
from cnn.ocr.network_config import init_imput_shape


def init_model(img_w, img_gen, ctc_lambda_func):
  """Initializes OCR network model
    Args:
      img_w - input image width
      img_get - image generator
      ctc_lambda_func - internal Keras loss function
    Returns:
      model - network model
  """
  
  input_shape = init_imput_shape(img_w)
  
  input_data = Input(name='the_input', shape=input_shape, dtype='float32')
  inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                        activation=act, init='he_normal', name='conv1')(input_data)
  inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
  inner = Convolution2D(conv_num_filters, filter_size, filter_size, border_mode='same',
                        activation=act, init='he_normal', name='conv2')(inner)
  inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

  conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_num_filters)
  inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

  # cuts down input size going into RNN:
  inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

  # Two layers of bidirecitonal GRUs
  # GRU seems to work as well, if not better than LSTM:
  gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(inner)
  gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(inner)
  gru1_merged = merge([gru_1, gru_1b], mode='sum')
  gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
  gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)

  # transforms RNN output to character activations:
  inner = Dense(img_gen.get_output_size(), init='he_normal',
                name='dense2')(merge([gru_2, gru_2b], mode='concat'))
  y_pred = Activation('softmax', name='softmax')(inner)
  Model(input=[input_data], output=y_pred).summary()

  labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
  input_length = Input(name='input_length', shape=[1], dtype='int64')
  label_length = Input(name='label_length', shape=[1], dtype='int64')
  # Keras doesn't currently support loss funcs with extra parameters
  # so CTC loss is implemented in a lambda layer
  loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

  # clipnorm seems to speeds up convergence
  model = Model(input=[input_data, labels, input_length, label_length], output=[loss_out])
  
  return (model, y_pred)

def init_input_data(input_shape):
  """Initializes network input by shape
    Args:
      input_shape - shape of input
    Returns:
      network input
  """
  return Input(name='the_input', shape=input_shape, dtype='float32')
