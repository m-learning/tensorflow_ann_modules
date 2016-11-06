"""
Created on Aug 15, 2016

Extracts layer from neural network

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.transfer.dataset_config import maybe_download_and_extract
import  cnn.transfer.graph_config as graph_config
import cnn.transfer.network_config as flags


class layer_features(object):
  """Class for network layer extraction"""
  
  def __init__(self, layer_name):
    self.layer_name = 'import/' + layer_name
    
  def extract_layer(self, tr_file):
    """Extracts network layer from network
      Args:
        tr_file - training file manager
      Returns:
        net_layer - Graph holding the trained Inception network, 
                    and various tensors we'll be manipulating.
    """
    
    flags.init_flags_only(tr_file)
    maybe_download_and_extract()
    net_layer = graph_config.get_layer(self.layer_name)
    
    return net_layer
