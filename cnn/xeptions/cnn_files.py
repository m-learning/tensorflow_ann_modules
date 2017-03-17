"""
Created on Jan 9, 2017

Files for training and evaluation data

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.utils.file_utils import cnn_file_utils


class training_file(cnn_file_utils):
  """Files and directories for (trained), 
     training, validation and test parameters"""
  
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('xeptions', image_resizer=image_resizer)
