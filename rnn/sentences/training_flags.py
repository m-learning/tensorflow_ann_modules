"""
Created on Mar 17, 2017

Parses command line argument and configures trainign and evaluation of the model 

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from rnn.sentences.rnn_files import training_file


_WEIGHTS_FILE_PATH = 'sentence_classifier_weights.h5'
_DEFAULT_LOSS_FUNCTION = 'categorical_crossentropy'
_BINARY_LOSS_FUNCTION = 'binary_crossentropy'

flags = None
_files = None

def _init_files():
  """Initializes file manager object
    Returns:
      _files - file manager object instance
  """
  
  if _files is None:
    global _files
    _files = training_file()
    
  return _files

def init_weights_path():
  """Initializes weights path to save and retrieve for network
    Returns:
      weights_path - path for serialized weights file
  """
  
  file_namager = _init_files()
  weights_path = file_namager.model_file(_WEIGHTS_FILE_PATH)
  
  return weights_path

def init_loss(args):
  """Initializes loss function
    Returns:
      loss - loss function name
  """
  
  if args is None:
      loss = _DEFAULT_LOSS_FUNCTION
  elif args.loss and args.loss == _BINARY_LOSS_FUNCTION:
    loss = _BINARY_LOSS_FUNCTION
  elif args.train_on_imdb:
    loss = _BINARY_LOSS_FUNCTION
  else:
    loss = _DEFAULT_LOSS_FUNCTION
    
  return loss

def parse_args():
  """Parses command line arguments
    Returns:
      args - command line arguments
  """

  parser = argparse.ArgumentParser()
  parser.add_argument('--max_review_length',
                      type=int,
                      default=500,
                      help='Maximum length of text')
  parser.add_argument('--embedding_vecor_length',
                      type=int,
                      default=32,
                      help='Embedding vector length')
  parser.add_argument('--top_words',
                      type=int,
                      default=5000,
                      help='Number words in data set')
  parser.add_argument('--loss',
                      type=str,
                      default=_DEFAULT_LOSS_FUNCTION,
                      help='Classification classes')
  parser.add_argument('--classes',
                      type=int,
                      default=28,
                      help='Number of classes classes')
  parser.add_argument('--train',
                      dest='train',
                      action='store_true',
                      help='Trains or evaluate model')
  parser.add_argument('--train_on_imdb',
                      dest='train_on_imdb',
                      action='store_true',
                      help='Trains on IMDB data set')
  parser.add_argument('--verbose',
                      dest='verbose',
                      action='store_true',
                      help='Log training or not')
  (args, _) = parser.parse_known_args()
  global flags
  flags = args
  
  return args
