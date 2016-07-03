'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import os

# Files and directory constant parameters
PATH_CNN_DIRECTORY = os.path.join('datas', 'mnist')
PATH_FOR_PARAMETERS = 'trained_data'
PATH_FOR_TRAINING = 'training_data'
WEIGHTS_FILE = 'conv_model.ckpt'

# Files and directories for parameters (trained), training, validation and test
class training_file:
  
    # Joins path from method
    def join_path(self, path_func, *other_path):
      
      result = None
      
      init_path = path_func()
      result = os.path.join(init_path, *other_path)
      
      return result
    
    # Gets current directory of script
    def get_current(self):
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        dirs = os.path.split(current_dir)
        dirs = os.path.split(dirs[0])
        current_dir = dirs[0]
        
        return current_dir
    
    # Gets directory for training set and parameters
    def get_data_directory(self):
        return self.join_path(self.get_current, PATH_CNN_DIRECTORY, PATH_FOR_TRAINING)
    
    # Initializes weights and biases files directory
    def init_files_directory(self):
        
        current_dir = self.join_path(self.get_current, PATH_CNN_DIRECTORY, PATH_FOR_PARAMETERS)
        
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        
        return current_dir
    
    # Gets training data  / parameters directory path
    def get_or_init_files_path(self):
        return self.join_path(self.init_files_directory, WEIGHTS_FILE)
