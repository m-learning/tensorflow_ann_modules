"""
Created on Jan 30, 2017

Runs VGGFaces implementation on Keras library

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy

from scipy import misc

from cnn.vgg import vgg_faces as faces
import numpy as np


def _parse_arguments():
  """Parses command line arguments
    Returns:
      flags - command line arguments
  """
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path',
                          type=str,
                          help='Path to face images.')
  parser.add_argument('--include_top',
                      dest='include_top',
                      action='store_true',
                      help='Include top layers')
  parser.add_argument('--not_include_top',
                      dest='include_top',
                      action='store_false',
                      help='Do not include top layers')
  parser.add_argument('--weights',
                      type=str,
                      default='vggface',
                      help='Weights for network')
  parser.add_argument('--input_tensor',
                      dest='input_tensor',
                      action='store_true',
                      help='Input tensor for network')
  parser.add_argument('--no_input_tensor',
                      dest='input_tensor',
                      action='store_false',
                      help='Do not specify input tensor for network')
  parser.add_argument('--nb_class',
                      type=int,
                      default=10,
                      help='Number of classes')
  (flags, _) = parser.parse_known_args()
  
  return flags

def _preprocess_image(flags):
  """Reads input image for network
    Args:
      flags - command line arguments
    Returns:
      img - binary image
  """
  im = misc.imread('../image/ak.jpg')
  im = misc.imresize(im, (224, 224)).astype(np.float32)
  aux = copy.copy(im)
  im[:, :, 0] = aux[:, :, 2]
  im[:, :, 2] = aux[:, :, 0]
  # Remove image mean
  im[:, :, 0] -= 93.5940
  im[:, :, 1] -= 104.7624
  im[:, :, 2] -= 129.1863
  im = np.expand_dims(im, axis=0)
  
  return im
  
if __name__ == '__main__':
  """Runs VGGFaces model for TensorFlow backend"""
  
  flags = _parse_arguments()
  model = faces.network_model(flags)
  im = _preprocess_image(flags)

  res = model.predict(im)
  
  if flags.include.include_top or not flags.input_tensor:
    print(np.argmax(res[0]))
  else:
    print(res)
