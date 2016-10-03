'''
Created on Aug 8, 2016

Files for object detection

@author: Levan Tsinadze
'''

from cnn.utils.file_utils import cnn_file_utils

HYPES_DIR = 'hypes'
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

# Files and directories for parameters (trained), training, validation and test
class training_file(cnn_file_utils):
  
  def __init__(self):
    super(cnn_file_utils, self).__init__('detect')
   
  # Gets path to trained parameters 
  def get_trained_files_dir(self):
    return self.init_files_directory()
  
  # Gets checkpoint file to restore trained parameters
  def get_checkpoint(self, iteration=190000):
    
    file_nm = 'save.ckpt-' + str(iteration)
    trined_dir = self.get_trained_files_dir()
    checkpoint_file = self.join_path(trined_dir, file_nm)
    
    return checkpoint_file
  
  # Gets hyperparameters configuration file
  def get_hypes_file(self):
    return self.join_path(self.get_data_general_directory, HYPES_DIR, 'overfeat_rezoom.json')
    
