"""
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cnn.utils.file_utils import files_and_path_utils


# Files and directory constant parameters
PATH_FOR_PARAMETERS = 'trained_data'
PATH_FOR_TRAINING = 'training_data'
WEIGHTS_FILE = 'conv_model.ckpt'

class training_file(files_and_path_utils):
  """Files and directories for parameters (trained), 
    training, validation and test"""
  
  def __init__(self):
    super(training_file, self).__init__('mnist')
      
  def get_current(self):
    """Gets current directory of script
      Retrun:
        current_dir - directory for training files
    """
      
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    dirs = os.path.split(current_dir)
    dirs = os.path.split(dirs[0])
    current_dir = dirs[0]
    
    return current_dir
  
  def get_data_directory(self):
    """Gets directory for training set and parameters
      Return:
        path to data directory
    """
    return self.join_path(self.get_current, self.path_to_cnn_directory, PATH_FOR_TRAINING)
  
  def init_files_directory(self):
    """Initializes weights and biases files directory
      Retuyrn:
        current_dir - path to initialized directory
    """
      
    current_dir = self.join_path(self.get_current, self.path_to_cnn_directory, PATH_FOR_PARAMETERS)
    
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    return current_dir
  
  def get_or_init_files_path(self):
    """Gets training data  / parameters directory path
      Return:
        path to model checkpoint file
    """
    return self.join_path(self.init_files_directory, WEIGHTS_FILE)
