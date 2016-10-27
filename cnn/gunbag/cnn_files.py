'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import glob
import os
import shutil

from cnn.utils.file_utils import cnn_file_utils

# Files and directory constant parameters
DATASET_DIR = '/home/levan-lev/Documents/ann/gunbag'

class training_file(cnn_file_utils):
  """Files and directories for parameters (trained), 
     training, validation and test"""
  
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('gunbag', image_resizer)
  
  def get_dataset_dir(self):
    """Method to get data set directory
      Return:
        data set directory
    """
    return super(training_file, self).get_training_directory()
    
  def convert_images(self, prfx, src_dir, dst_dir, img_type):
    """Converts passed images for training and recognition
      Args:
        prfx - image prefix
        src_dir - source directory
        dst_dir - destination directory
        img_type - image type
    """
    
    i = 0
    scan_persons_dir = os.path.join(src_dir, img_type)
    for pr in glob.glob(scan_persons_dir):
      fl_name = prfx + 'cnvrt_data_' + str(i) + '.jpg'
      n_im = os.path.join(dst_dir, fl_name)
      if not os.path.exists(n_im):
        self.read_and_write(pr, n_im)
        os.remove(pr)
      i += 1
  
  def get_dataset_jpg_dir(self, dest_directory, zip_ref):
    """Gets training training set
      Args:
        dest_directory - destination directory
        zip_ref - archive file reference
      Return:
        dataset_dir - dataset directory
    """
    
    dataset_dir = os.path.join(dest_directory , DATASET_DIR)
    
    if os.path.exists(dataset_dir):
      shutil.rmtree(dataset_dir, ignore_errors=True)
    zip_ref.extractall(dest_directory)
    
    return dataset_dir
  
  def get_or_init_training_set(self):
    """Gets or generates training set"""
    self.get_or_init_files_path()
    self.get_or_init_data_directory()
    print 'Training set is prepared'
