"""
Created on Nov 8, 2016
Flags for VGG network model training and evaluation
@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


trainign_flags = None

def parse_and_retrieve():
  """Retrieves command line arguments"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--training_steps',
                          default=150000,
                          help='Number of training iterations',
                          type=int)
  arg_parser.add_argument('--keep_prob',
                          default=0.5,
                          help='Dropout keep probability',
                          type=float) 
  (argument_flags, _) = arg_parser.parse_known_args()
  global trainign_flags
  trainign_flags = argument_flags
