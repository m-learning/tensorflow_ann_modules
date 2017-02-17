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

def init_imput_shape(img_w):
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
