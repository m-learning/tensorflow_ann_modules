"""
Created on Jan 12, 2017

Utility module for FaceNet model

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

GRAPH_FILE = 'face_embeddings.pb'

INPUT_NAME = 'input'

INPUT_LAYER = 'input:0'
TRAIN_LAYER = 'phase_train:0'
EMBEDDINGS_LAYER = 'embeddings:0'
