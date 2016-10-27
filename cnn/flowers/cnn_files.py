'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import glob
import os
import shutil
import sys
import tarfile

from cnn.utils.file_utils import cnn_file_utils
from six.moves import urllib


# Files and directory constant parameters
TRAINIG_SET_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

class training_file(cnn_file_utils):
  """Files and directories for parameters (trained), 
     training, validation and test"""
  
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('flowers', image_resizer)
    
  def get_dataset_dir(self):
    """Method to get data set directory
      Return:
        data set directory
    """
    return super(training_file, self).get_data_directory()
  
  def resize_flower_images(self, training_dir):
    """Resizes flower images
      Args:
        training_dir - training files directory
    """
    
    if self.image_resizer:
      scan_dir = self.join_path(training_dir, 'flower_photos')
      if os.path.exists(scan_dir):
        flower_dirs = ('daisy', 'dandelion', 'tulips', 'roses', 'sunflowers')
        for scan_sub_dir in flower_dirs:
          flower_dir_pref = self.join_path(scan_dir, scan_sub_dir)
          if os.path.exists(flower_dir_pref):
            flower_dir = os.path.join(flower_dir_pref, '*.jpg')
            for pr in glob.glob(flower_dir):
              self.read_and_write(pr, pr)
      
  def get_or_init_training_set(self):
    """Gets or generates training set"""
    
    dest_directory = self.get_archives_directory()
    filename = TRAINIG_SET_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(TRAINIG_SET_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    training_dir = self.get_training_directory()
    if not os.path.exists(training_dir):
      os.mkdir(training_dir)  
    else:
      shutil.rmtree(training_dir, ignore_errors=True)
      os.mkdir(training_dir)
    tarfile.open(filepath, 'r:gz').extractall(training_dir)
    self.resize_flower_images(training_dir)
