"""
Created on Nov 25, 2016
Resizes images for interface
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.transfer.recognizer_interface import retrained_recognizer

def resize_for_interface(image_data):
  
  class document_image_recognizer(retrained_recognizer):
    """Class for document image recognizer"""