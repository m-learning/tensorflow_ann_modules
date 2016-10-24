'''
Created on Aug 15, 2016

Extracts layer from neural network

@author: Levan Tsinadze
'''

from cnn.transfer.config_image_net import maybe_download_and_extract
import cnn.transfer.config_image_net as config
import  cnn.transfer.graph_config as graph_config


# Layer extractor from neural network
class layer_features(object):
  """Class for network layer extraction"""
  
  def __init__(self, layer_name):
    self.layer_name = 'import/' + layer_name
    
  def extract_layer(self, tr_file):
    """Extracts network layer
      Args:
        tr_file - training file manager
    """
    
    tr_flags = config.init_flags_only(tr_file)
    maybe_download_and_extract()
    graph_config.get_layer(tr_flags, self.layer_name)