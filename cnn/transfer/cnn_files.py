"""
Created on Nov 13, 2016
Transfer learning files manager
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

from cnn.utils.file_utils import cnn_file_utils


class training_file(cnn_file_utils):
  """Files and directories for (trained), 
     training, validation and test parameters"""
  
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('transfer', image_resizer=image_resizer)
  
  def get_dataset_dir(self):
    """Method to get data set directory
      Returns:
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
    
  def get_or_init_training_set(self):
    """Gets or generates training set"""
    self.get_or_init_files_path()
    self.get_or_init_data_directory()
    print('Training set is prepared')
