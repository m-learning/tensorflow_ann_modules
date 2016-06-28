'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import os

PATH_CNN_DIRECTORY = '/datas/mnist/'
PATH_FLOWER_DIRECTORY = '/datas/mnist/'
PATH_FOR_PARAMETERS = 'trained_data/'
PATH_FOR_TRAINING = 'training_data/'
WEIGHTS_FILE = 'conv_model.ckpt'

class training_file:
    
    # Gets current directory of script
    def get_current(self):
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        dirs = os.path.split(current_dir)
        dirs = os.path.split(dirs[0])
        current_dir = dirs[0]
        
        return current_dir
    
    # Gets directory for training set and parameters
    def get_data_directory(self):
        
        current_dir = self.get_current()
        
        current_dir += PATH_CNN_DIRECTORY 
        current_dir += PATH_FOR_TRAINING
        
        return current_dir
    
    def init_files_directory(self):
        
        current_dir = self.get_current()
        
        current_dir += PATH_CNN_DIRECTORY 
        current_dir += PATH_FOR_PARAMETERS
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        
        return current_dir
    
    # Gets training data  / parameters directory path
    def get_or_init_files_path(self):
        
        current_dir = self.init_files_directory()
        current_dir += WEIGHTS_FILE
        
        return current_dir
        
        


print training_file().get_or_init_files_path()
