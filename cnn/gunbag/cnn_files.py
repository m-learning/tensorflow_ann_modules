'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import glob
import os
import shutil

from cnn.utils.file_utils import cnn_file_utils


DATASET_DIR = '/home/levan-lev/Documents/ann/gunbag'

# Files and directory constant parameters

# Files and directories for parameters (trained), training, validation and test
class training_file(cnn_file_utils):
  
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('gunbag', image_resizer)
  
  # Method to get data set directory
  def get_dataset_dir(self):
    return super(training_file, self).get_training_directory()
    
  # Converts person images
  def convert_images(self, prfx, src_dir, persons_dir, img_type):
    
    i = 0
    scan_persons_dir = os.path.join(src_dir, img_type)
    for pr in glob.glob(scan_persons_dir):
      fl_name = prfx + 'cnvrt_data_' + str(i) + '.jpg'
      n_im = os.path.join(persons_dir, fl_name)
      if not os.path.exists(n_im):
        self.read_and_write(pr, n_im)
        os.remove(pr)
      i += 1
  
  # Gets persons training set
  def get_dataset_jpg_dir(self, dest_directory, zip_ref):
    
    pers_dir = os.path.join(dest_directory , DATASET_DIR)
    
    if os.path.exists(pers_dir):
      shutil.rmtree(pers_dir, ignore_errors=True)
    zip_ref.extractall(dest_directory)
    
    return pers_dir
  
  # Gets or generates training set
  def get_or_init_training_set(self):
    self.get_or_init_files_path()
    self.get_or_init_data_directory()
    print 'Training set is prepared'
