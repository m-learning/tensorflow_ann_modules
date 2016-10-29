"""
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import shutil

from cnn.utils.file_utils import cnn_file_utils


# Files and directory constant parameters
PERSONS_SETS = ['/storage/ann/crossings_set/empty_crossings', '/storage/ann/crossings_set/person_crossings']
TRAINIG_ZIP_FOLDER = 'training_arch'
CASSIFICATION_DIRS = ['empty_crossings', 'person_crossings']

# Directories to move training data from
EMPTY_CROSSINGS_DIR = 'empty_crossings'
PERSON_CROSSINGS_DIR = 'person_crossings'

class training_file(cnn_file_utils):
  """Files and directories for parameters (trained), training, validation and test"""
  
  def __init__(self):
    super(training_file, self).__init__('zebras')
    
  def copy_crosswalk_images(self, src_dir, dst_dir, img_type):
    """Converts person images with cross walks
      Args:
        src_dir - source directory
        dst_dir - destination directory
        img_type - image type to copy
    """
    
    scan_persons_dir = self.join_path(src_dir, img_type)
    for pr in glob.glob(scan_persons_dir):
      shutil.copy(pr, dst_dir)
      
  def get_crosswalk_set(self):
    """Gets persons data set"""
    
    training_dir = self.get_data_directory()
    for i in  range(len(PERSONS_SETS)):
      src_dir = PERSONS_SETS[i]
      dst_dir = self.init_dir(training_dir, CASSIFICATION_DIRS[i])
      self.copy_crosswalk_images(src_dir, dst_dir, '*.jpg')      
    
  
  def get_or_init_training_set(self):
    """Gets or generates training set"""
    self.get_crosswalk_set()
    print('Training set is prepared')
