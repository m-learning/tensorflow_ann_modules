"""
Created on Oct 19, 2016

Image size utility for VGG networks 

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.utils.image_resizing import image_resizer


vgg_dim = 224
vgg_size = (vgg_dim, vgg_dim)

class vgg_image_resizer(image_resizer):
  """Resizes image for VGG network"""
  
  def __init__(self):
    super(vgg_image_resizer, self).__init__(vgg_dim, vgg_dim)
