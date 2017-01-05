"""
Created on Jan 4, 2017

Files manager module for Inception training sets

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from cnn.utils import file_utils as _files
from cnn.utils.file_utils import cnn_file_utils
from six.moves import urllib


TRAIN_URL = 'http://pjreddie.com/media/files/mnist_train.csv'
TEST_URL = 'http://pjreddie.com/media/files/mnist_test.csv'

TRAINING_SET = 'mnist_train.csv'
TEST_SET = 'mnist_test.csv'

CHECKPOINT_FILE = 'mnist_model.ckpt'

class training_file(cnn_file_utils):
  """Files and directories for (trained), 
     training, validation and test parameters"""
     
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('inceptions', image_resizer=image_resizer)
    
  def get_dataset_dir(self):
    """Method to get data set directory
      Returns:
        data set directory
    """
    return super(training_file, self).get_training_directory()
  
  def _get_data_file_path(self, filename):
    """Gets training or test set CSV file path
      Args:
        filename - file name
      Returns:
        filepath - training or test set file path
    """
    dest_directory = self.get_dataset_dir()
    _files.ensure_dir_exists(dest_directory)
    filepath = self.join_path(dest_directory, filename)
    
    return filepath
  
  def _download_data_files(self, filename, url_path):
    """Downloads training and test CSV files
      Args:
        filename - file name
        url_path - file URL address
    """
    
    filepath = self._get_data_file_path(filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(url_path, filepath, _progress)
    print("Get statinfo")
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  
  def get_or_init_training_set(self):
    """Gets or generates training set"""
    
    self._download_data_files(TRAINING_SET, TRAIN_URL)
    self._download_data_files(TEST_SET, TEST_URL)
    
  def _init_trained_parameters(self):
    """Gets trained parameters checkpoint if exists
      Returns:
        checkpoint_path - model checkpoint file path
    """
    
    trained_files_dir = self.init_files_directory()
    checkpoint_path = self.join_path(trained_files_dir, CHECKPOINT_FILE)
    
    return checkpoint_path
  
  @property
  def checkpoint_path(self):
    """Gets trained parameters checkpoint if exists
      Returns:
        model checkpoint file path
    """
    
    return self._init_trained_parameters()    
  
  @property
  def training_set(self):
    """Gets training set CSV file path
      Returns:
        training set file path
    """
    return self._get_data_file_path(TRAINING_SET)
  
  @property
  def test_set(self):
    """Gets test set CSV file path
      Returns:
        test set file path
    """
    return self._get_data_file_path(TEST_SET)
  