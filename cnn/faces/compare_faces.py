"""
Created on Jan 12, 2017

Compares faces throw FaceNet model
Performs face alignment and calculates L2 distance between the embeddings of two images.

@author: Levan Tsinadze

MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.faces import face_utils as utils
from cnn.faces import network_interface as interface
from cnn.faces.cnn_files import training_file
import numpy as np


def compare_faces(args):
  """Generates many face embeddings from files and calculates L2 distances
    Args:
      args - command line arguments
    Returns:
      tuple of
        emb - embeddings
        nrof_images - number of images
  """

  images = utils.load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction, _files)
  emb = interface.calculate_embeddings(args.model_dir, images)
          
  nrof_images = len(args.image_files)

  print('Images:')
  for i in range(nrof_images):
    print('%1d: %s' % (i, args.image_files[i]))
  print('')
  
  # Print distance matrix
  print('Distance matrix')
  print('    ', end='')
  
  return (emb, nrof_images)

def compare_many_faces(args):
  """Generates many face embeddings from files and calculates L2 distances
    Args:
      args - command line arguments
  """

  (emb, nrof_images) = compare_faces(args)
  for i in range(nrof_images):
    print('    %1d     ' % i, end='')
  print('')
  for i in range(nrof_images):
    print('%1d  ' % i, end='')
    for j in range(nrof_images):
      dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
      print('  %1.4f  ' % dist, end='')
    print('')
      
def compare_two_faces(args):
  """Generates two face embeddings from files and calculates L2 distances
    Args:
      args - command line arguments
  """

  (emb, _) = compare_faces(args)
  print('')
  dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))
  print('  %1.4f  ' % dist, end='')
  print('')

def parse_arguments():
  """Parses command line arguments
    Returns:
      argument_flags - retrieved arguments
  """
  global _files
  _files = training_file()
  parser = argparse.ArgumentParser()
  parser.add_argument('--many_faces',
                       dest='many_faces',
                       action='store_true',
                       help='Flag to compare only two faces.')
  parser.add_argument('--two_faces',
                       dest='many_faces',
                       action='store_false',
                       help='Do not print data set file names and labels.')
  parser.add_argument('--model_dir',
                      type=str,
                      default=_files.model_dir,
                      help='Directory containing the meta_file and ckpt_file')
  parser.add_argument('--image_files',
                      type=str,
                      nargs='+',
                      help='Images to compare')
  parser.add_argument('--image_size',
                      type=int,
                      default=160,
                      help='Image size (height, width) in pixels.')
  parser.add_argument('--margin',
                      type=int,
                      default=44,
                      help='Margin for the crop around the bounding box (height, width) in pixels.')
  parser.add_argument('--gpu_memory_fraction',
                      type=float,
                      default=1.0,
                      help='Upper bound on the amount of GPU memory that will be used by the process.')
  parser.add_argument('--result_file',
                      type=str,
                      help='File to store compared result.')
  (argument_flags, _) = parser.parse_known_args()
  
  return argument_flags

if __name__ == '__main__':
  """Compares face embeddings from image files"""
  
  argument_flags = parse_arguments()
  if argument_flags.many_faces:
    compare_many_faces(argument_flags)
  else:
    compare_two_faces(argument_flags)
