"""
Created on Aug 27, 2017

Network configuration parameters

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from cnn.famnist.cnn_files import training_file

_MODEL_FILE = 'fmnist.md5'
_WEIGHTS_FILE = 'fmnist.h5'

def _init_config():
  """Initializes configuration arguments parser
    Returns:
      tuple of -
        _files - file manager
        arg_parser - arguments parser
  """
  
  _files = training_file()
  arg_parser = argparse.ArgumentParser()
  
  return (_files, arg_parser)


def _add_common_parameters(_files, arg_parser):
  """Adds common parameters to configuration flags
    Args:
      arg_parser - configuration parameters parser
      _files - files manager
  """
  
  # Saved weights and network model
  arg_parser.add_argument('--model',
                          type=str,
                          default=_files.model_file(_MODEL_FILE),
                          help='Where to save the trained data')
  arg_parser.add_argument('--weights',
                          type=str,
                          default=_files.model_file(_WEIGHTS_FILE),
                          help='Where to save the trained model')
  # Network configuration
  arg_parser.add_argument('--num_classes',
                          type=int,
                          default=10,
                          help='Number of output classes')
  arg_parser.add_argument('--image_height',
                          type=int,
                          default=28,
                          help='Input image height')
  arg_parser.add_argument('--image_width',
                          type=int,
                          default=28,
                          help='Input image width')

def read_interface_parameters():
  """Parses and retrieves parameters to run the interface
    Returns:
      arg_parser - argument parses
      flags - configuration flags
  """
  
  (_files, arg_parser) = _init_config()
  # Network configuration
  _add_common_parameters(_files, arg_parser)
  # Host and port for HTTP service
  arg_parser.add_argument('--host',
                          type=str,
                          default='0.0.0.0',
                          help='Host name for HTTP service')
  arg_parser.add_argument('--port',
                          type=int,
                          default=8080,
                          help='Port number for HTTP service')
  (flags, _) = arg_parser.parse_known_args()
  
  return (flags, arg_parser)


def read_training_parameters():
  """Reads command line arguments
    Returns:
      flags - training flags
  """
  
  (_files, arg_parser) = _init_config()
  # Training hyper-parameters
  arg_parser.add_argument('--epochs',
                          type=int,
                          default=120,
                          help='Number of training epochs')
  arg_parser.add_argument('--batch_size',
                          type=int,
                          default=128,
                          help='Training batch size')
  # Network configuration
  _add_common_parameters(_files, arg_parser)

  (flags, _) = arg_parser.parse_known_args()
  
  return flags
