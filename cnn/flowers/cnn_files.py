'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import os
import shutil
import sys
import tarfile

from cnn.utils.file_utils import cnn_file_utils
from six.moves import urllib


# Files and directory constant parameters
TRAINIG_SET_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# Files and directories for parameters (trained), training, validation and test
class training_file(cnn_file_utils):
  
  def __init__(self):
    super(training_file, self).__init__('flowers')
    
    # Method to get data set directory
  def get_dataset_dir(self):
    return super(training_file, self).get_data_directory()
      
  # Gets or generates training set
  def get_or_init_training_set(self):
    
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
