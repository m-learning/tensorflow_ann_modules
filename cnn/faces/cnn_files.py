"""
Created on Jan 9, 2017

Files for training and evaluation data

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.faces.face_utils import GRAPH_FILE
from cnn.utils.file_utils import cnn_file_utils


class training_file(cnn_file_utils):
  """Files and directories for (trained), 
     training, validation and test parameters"""
  
  def __init__(self, image_resizer=None):
    super(training_file, self).__init__('faces', image_resizer=image_resizer)
    
  @property
  def pairs_file(self):
    """Gets evaluation face pairs file path
      Returns:
        pairs_file - pairs file path
    """
    
    eval_dir = self.eval_dir
    pairs_file = self.join_and_init_path(eval_dir, 'pairs.txt')
    
    return pairs_file
  
  @property
  def graph_file(self):
    """Gets graph file path
      Returns:
        graph_path - file path for model's graph
    """
    return self.join_path(self.model_dir, GRAPH_FILE)
