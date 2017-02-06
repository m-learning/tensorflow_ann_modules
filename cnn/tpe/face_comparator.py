"""
Created on Feb 6, 2017

Runs face comparator as service

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.tpe.compare_faces import init_verificator, compare_faces


dist = 0.85

class Images:
  """Image container class"""
  
  def __init__(self):
    self.image1 = None
    self.image2 = None
    self.score = True

if __name__ == '__main__':
  """Generates tensors from images"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--score',
                          dest='score',
                          action='store_true',
                          help='Flags for face embedding compare.')
  (command_args, _) = arg_parser.parse_known_args()
  fv = init_verificator()
  flags = Images()
  flags.score = command_args.score
  while True:
    flags.image1 = raw_input('Input image1 path: ')
    flags.image2 = raw_input('Input image2 path: ')
    if flags.image1 and flags.image2:
      comp_result = compare_faces(flags, fv)
    else:
      print('No images to compare')
