"""
Created on Jul 6, 2016

Utility class for training test and validation data files

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import types


try:
  from PIL import Image
except ImportError:
  print("Importing Image from PIL threw exception")
  import Image


# General parent directory for files
DATAS_DIR_NAME = 'datas'

# Training set archives directory suffix
TRAINIG_ZIP_FOLDER = 'training_arch'

# Files and directory constant parameters
PATH_FOR_PARAMETERS = 'trained_data'
PATH_FOR_TRAINING = 'training_data'
PATH_FOR_EVALUATION = 'eval_data'
PATH_FOR_TRAINING_PHOTOS = 'flower_photos'
WEIGHTS_FILE = 'output_graph.pb'
LABELS_FILE = 'output_labels.txt'

# Test files and directories
TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_NAME = 'test_image'

# Counts files in directory
def count_files(dir_name):
  
  file_count = 0
  for _, _, files in os.walk(dir_name):
    file_count += len(files)
  
  return file_count

class files_and_path_utils(object):
  """Utility class for files and directories"""
  
  def __init__(self, parent_cnn_dir, path_to_training_photos=None):
    self.path_to_cnn_directory = os.path.join(DATAS_DIR_NAME, parent_cnn_dir)    
    if path_to_training_photos is None:
      self.path_to_training_photos = PATH_FOR_TRAINING_PHOTOS
    else:
      self.path_to_training_photos = path_to_training_photos
  
    # Creates file if not exists
  def init_file_or_path(self, file_path):
    
    if not os.path.exists(file_path):
      os.makedirs(file_path)
    
    return file_path
  
  # Joins path from method
  def join_path(self, path_inst, *other_path):
    """Joins passed file paths and function generating path
      Args:
       path_inst file path or function
      Returns:
       generated file path 
    """
    if isinstance(path_inst, types.StringType):
      init_path = path_inst
    else:
      init_path = path_inst()
    result = os.path.join(init_path, *other_path)
    
    return result
  
  def join_and_init_path(self, path_inst, *other_path):
    """Joins and creates file or directory paths
      Args:
        path_inst - image path or function 
                    returning path
        other_path - vavargs for other paths
                     or functions
      Return:
        result - joined path
    """
    
    result = self.join_path(path_inst, *other_path)
    self.init_file_or_path(result)
    
    return result
  
  def init_dir(self, dir_path, *other_path):
    """Creates appropriated directory 
       if such does not exists
      Args:
        dir_path - directory path
        *other_path - vavargs for other paths
                     or functions
      Return:
        result_dir - joined directory path
    """
    
    result_dir = self.join_path(dir_path, *other_path)
    self.init_file_or_path(result_dir)
    
    return result_dir 
  
  def get_current(self):
    """Gets current directory of script
      Return:
        current_dir - current directory
    """
      
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    dirs = os.path.split(current_dir)
    dirs = os.path.split(dirs[0])
    current_dir = dirs[0]
    
    return current_dir
  

class cnn_file_utils(files_and_path_utils):
  """Utility class for training and testing files and directories"""
  
  def __init__(self, parent_cnn_dir, image_resizer=None):
    super(cnn_file_utils, self).__init__(parent_cnn_dir)
    self.image_resizer = image_resizer
    
    # Reads image with or without resizing
  def read_image(self, pr):
    
    if self.image_resizer is None:
      im = Image.open(pr)
    else:
      im = self.image_resizer.read_and_resize(pr)
      
    return im
  
  # Writes image with or without resizing
  def write_image(self, im, n_im):
    
    if self.image_resizer is None:
      im.save(n_im)
    else:
      self.image_resizer.save_resized(im, n_im)
      
  # Reads and saves (resized or not) image from one path to other
  def read_and_write(self, pr, n_im):
    
    if self.image_resizer is None:
      im = Image.open(pr)
      im.save(n_im)
    else:
      self.image_resizer.read_resize_write(pr, n_im)
  
  # Gets or creates directories
  def get_data_general_directory(self):
    return self.join_and_init_path(self.get_current, self.path_to_cnn_directory)
  
  # Gets training set archives directory
  def get_archives_directory(self):
    
    dest_directory = self.join_path(self.get_data_general_directory, TRAINIG_ZIP_FOLDER)
    if not os.path.exists(dest_directory):
      os.mkdir(dest_directory) 
    return dest_directory
  
  # Gets training data directory
  def get_training_directory(self):
    return self.join_path(self.get_data_general_directory, PATH_FOR_TRAINING)

  # Gets directory for training set and parameters
  def get_data_directory(self):
    return self.join_path(self.get_training_directory, self.path_to_training_photos)
  
  # Gets or creates directory for training set and parameters
  def get_or_init_data_directory(self):
    
    dir_path = self.get_data_directory()
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
  
  # Gets or creates directory for trained parameters
  def init_files_directory(self):
      
    current_dir = self.join_path(self.get_data_general_directory, PATH_FOR_PARAMETERS)
    
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    return current_dir

  # Initializes trained files path
  def get_or_init_files_path(self):
    return self.join_path(self.init_files_directory, WEIGHTS_FILE)
      
  # Gets training data  / parameters directory path
  def get_or_init_labels_path(self):
    return self.join_path(self.init_files_directory, LABELS_FILE)

  # Gets directory for test images
  def get_or_init_test_dir(self):
    
    current_dir = self.join_path(self.get_data_general_directory, TEST_IMAGES_DIR)
    
    if not os.path.exists(current_dir):
      os.mkdir(current_dir)  
    
    return current_dir
    
  # Gets or initializes test image
  def get_or_init_test_path(self):
    return self.join_path(self.get_or_init_test_dir, TEST_IMAGE_NAME)
  
  # Gets / initializes evaluation directory
  def get_or_init_eval_path(self):
    return self.join_and_init_path(self.get_data_general_directory, PATH_FOR_EVALUATION)
