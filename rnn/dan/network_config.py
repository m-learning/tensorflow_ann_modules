"""
Created on Jan 27, 2017

Configuration parameters for DAN implementation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 150
nb_epoch = 2
dropout = 0.2
nb_hidden_layers = 3

# Trained weights file name
_WEIGHTS_FILE = 'dan_weights.h5'

def parse_arguments():
  """Parses command line arguments
    Returns:
      args - parsed arguments
  """
  
  parse_args = argparse.ArgumentParser()
  parse_args.add_argument('--epochs',
                          type=int,
                          default=nb_epoch,
                          help='Number of training epochs')
  (args, _) = parse_args.parse_known_args()
  
  return args
