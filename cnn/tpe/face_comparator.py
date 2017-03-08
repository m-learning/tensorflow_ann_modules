"""
Created on Feb 6, 2017

Runs face comparator as service

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.tpe import compare_faces as comparator
from cnn.tpe.compare_faces import init_verificator, add_arguments, compare_faces


dist = 0.85

class Images:
  """Image container class"""
  
  def __init__(self):
    self.image1 = None
    self.image2 = None
    self.score = True
    self.output1 = None
    self.output2 = None
    
def run_comparation(flags, fv):
  """Runs image comparation
    Args:
      flags - image paths and command line flags
  """
  if flags.image1 and flags.image2:
    compare_faces(flags, fv)
  else:
    print('No images to compare')

if __name__ == '__main__':
  """Generates tensors from images"""
  
  _files = comparator._files
  eval_dir = _files.eval_dir
  arg_parser = argparse.ArgumentParser()
  command_args = add_arguments(arg_parser)
  fv = init_verificator()
  flags = Images()
  flags.score = command_args.score
  flags.output1 = command_args.output1
  flags.output2 = command_args.output2
  while True:
    flags.image1 = raw_input('Input image1 path: ')
    flags.image2 = raw_input('Input image2 path: ')
    run_comparation(flags, fv)
