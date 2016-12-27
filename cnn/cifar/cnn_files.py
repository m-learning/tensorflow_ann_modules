"""
Created on Dec 21, 2016

Files for training data

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.utils.file_utils import cnn_file_utils


PATH_FOR_LOGS = 'logs'

PATH_FOR_LOG_FILES = 'outputlog.out'

PATH_FOR_LOG_ERRORS = 'errorlog.out'

class training_file(cnn_file_utils):
  """Files and directories for (trained), 
     training, validation and test parameters"""
     
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('cifar', image_resizer=image_resizer)
    
  def init_logs_directory(self):
    """Gets or creates directory for logs
      Returns:
        directory for log files
    """
      
    return self.join_and_init_path(self.get_data_general_directory, PATH_FOR_LOGS)
  
  def init_log_files(self):
    """Gets or creates path to output log files
      Returns:
        log_files - output log file path    
    """  
    
    dir_path = self.init_logs_directory()
    log_files = self.join_path(dir_path, PATH_FOR_LOG_FILES)
    
    return log_files
  
  def init_error_files(self):
    """Gets or creates path to error log files
      Returns:
        log_errors - error log file path    
    """ 
    
    dir_path = self.init_logs_directory()
    log_errors = self.join_path(dir_path, PATH_FOR_LOG_ERRORS)
    
    return log_errors
