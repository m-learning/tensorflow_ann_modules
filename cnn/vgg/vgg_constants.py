"""
Created on Oct 12, 2016
Constant parameters for VGG network implementation
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

trainable_scopes = 'vgg_16/fc6,vgg_16/fc7,vgg_16/fc8'  # Trainable layers
checkpoint_exclude_scopes = trainable_scopes  # Layers to be excluded during the training
checkpoint_file = 'vgg_16'  # Checkpoint file name
checkpoint_url = 'vgg_16_2016_08_28'  # Checkpoint URL address
