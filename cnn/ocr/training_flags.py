"""
Created on Feb 28, 2017

Flags for model training and evaluation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime


def parse_arguments():
  """Parses command line arguments
    Returns:
      args - parsed command line arguments
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_width',
                      type=int,
                      default=128,
                      help='Input image width')
  parser.add_argument('--start_epoch',
                      type=int,
                      default=0,
                      help='Training start epoch')
  parser.add_argument('--stop_epoch',
                      type=int,
                      default=20,
                      help='Training stop epoch')
  parser.add_argument('--Run name',
                      type=str,
                      default=datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                      help='Training run name')
  (args, _) = parser.parse_known_args()
  
  return args
