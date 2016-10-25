"""
Created on Jun 28, 2016

Runs retrained neural network for recognition

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from cnn.gunbag.cnn_files import training_file
from cnn.transfer.general_recognizer import retrained_recognizer


# Recognizes image thru trained neural networks
class image_recognizer(retrained_recognizer):
  
  def __init__(self):
    tr_file = training_file()
    super(image_recognizer, self).__init__(tr_file)

if __name__ == '__main__':
  """Runs image recognition"""
  
  img_recognizer = image_recognizer()
  img_recognizer.run_inference_on_image(sys.argv)
