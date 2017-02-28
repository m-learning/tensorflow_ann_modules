"""
Created on Feb 18, 2017

Network configuration for OCR module

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

from cnn.ocr.cnn_files import training_file
from cnn.ocr.image_ocr_keras import TextImageGenerator


img_h = 64
conv_num_filters = 16
filter_size = 3
pool_size = 2
rnn_size = 512
time_dense_size = 32
act = 'relu'

words_per_epoch = 16000
val_split = 0.2
val_words = int(words_per_epoch * (val_split))

_files = training_file()
OUTPUT_DIR = _files.model_dir
DATA_DIR = _files.data_dir

def init_conv_to_rnn_dims(img_w):
  """Initializes dimentions for RNN conversion
    Args:
      img_w
    Returns:
      generated dimensions
  """
  
  return (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_num_filters)

def init_input_shape(img_w):
  """Initializes image input shape
    Args:
      img_w - image width
    Returns:
      input_shape - input image tensor shape
  """
  
  if K.image_dim_ordering() == 'th':
    input_shape = (1, img_w, img_h)
  else:
    input_shape = (img_w, img_h, 1)
    
  return input_shape

def init_img_gen(img_w):
  """Image generator from text and words
    Args:
      img_w - image width
    Returns:
      img_gen - image generator object
  """
  img_gen = TextImageGenerator(monogram_file=_files.data_file('wordlist_mono_clean.txt'),
                               bigram_file=_files.data_file('wordlist_bi_clean.txt'),
                               minibatch_size=32,
                               img_w=img_w,
                               img_h=img_h,
                               downsample_factor=(pool_size ** 2),
                               val_split=words_per_epoch - val_words)
  return img_gen

def ctc_lambda_func(args):
  """Lambda for CTC input
    Args:
      args - arguments
    Returns:
      result_fnc - result function
  """
 
  (y_pred, labels, input_length, label_length) = args
  # the 2 is critical here since the first couple outputs of the RNN
  # tend to be garbage:
  y_pred = y_pred[:, 2:, :]
  result_fnc = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
  
  return result_fnc
