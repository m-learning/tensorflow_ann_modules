"""
Created on Mar 17, 2017

Parses command line argument and configures trainign and evaluation of the model 

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rnn.sentences.rnn_files import training_file


_WEIGHTS_FILE_PATH = 'sentence_classifier_weights.h5'

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
