"""
Created on Feb 15, 2017

Faces comparator based on DLIB faces

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.faces import dlib_compare_faces as comparator 


def _parse_arguments():
  """Parses command line arguments
    Returns:
      args - parsed command line arguments
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--image1',
                      type=str,
                      help='Path to first image')
  parser.add_argument('--image2',
                      type=str,
                      help='Path to second image')
  parser.add_argument('--include_gui',
                      dest='include_gui',
                      action='store_true',
                      help='Include top layers')
  parser.add_argument('--verbose',
                      dest='verbose',
                      action='store_true',
                      help='Print additional information')
  (args, _) = parser.parse_known_args()
  
  return args

if __name__ == '__main__':
  """Starts face comparator service"""
  
  args = _parse_arguments()
  _network = comparator.load_model()
  while True:
    image1 = raw_input('Input image1 path: ')
    image2 = raw_input('Input image2 path: ')
    face_dists = comparator.compare_files(image1, image2, _network)
    comparator.print_faces(face_dists)