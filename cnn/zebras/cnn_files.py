'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import shutil
import glob

from cnn.utils.file_utils import cnn_file_utils

# Files and directory constant parameters
PERSONS_SETS = ['/storage/ann/crossings_set/empty_crossings', '/storage/ann/crossings_set/person_crossings']
TRAINIG_ZIP_FOLDER = 'training_arch'
CASSIFICATION_DIRS = ['empty_crossings', 'person_crossings']

# Directories to move training data from
EMPTY_CROSSINGS_DIR = 'empty_crossings'
PERSON_CROSSINGS_DIR = 'person_crossings'

# Files and directories for parameters (trained), training, validation and test
class training_file(cnn_file_utils):
  
  def __init__(self):
    super(training_file, self).__init__('zebras')
    
  # Converts person images with cross walks
  def copy_crosswalk_images(self, src_dir, dst_dir, img_type):
    
    scan_persons_dir = self.join_path(src_dir, img_type)
    for pr in glob.glob(scan_persons_dir):
      shutil.copy(pr, dst_dir)
      
  # Gets persons data set
  def get_crosswalk_set(self):
    
    training_dir = self.get_data_directory()
    for i in  range(len(PERSONS_SETS)):
      src_dir = PERSONS_SETS[i]
      dst_dir = self.init_dir(training_dir, CASSIFICATION_DIRS[i])
      self.copy_crosswalk_images(src_dir, dst_dir, '*.jpg')      
    
  
  # Gets or generates training set
  def get_or_init_training_set(self):
    self.get_crosswalk_set()
    print 'Training set is prepared'
