"""
Created on Jan 30, 2017

Runs VGGFaces implementation on Keras library

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.vgg.vgg_faces import VGGFace
import numpy as np

if __name__ == '__main__':
  from scipy import misc
  import copy
  # tensorflow
  model = VGGFace()
  im = misc.imread('../image/ak.jpg')
  im = misc.imresize(im, (224, 224)).astype(np.float32)
  aux = copy.copy(im)
  im[:, :, 0] = aux[:, :, 2]
  im[:, :, 2] = aux[:, :, 0]
  # Remove image mean
  im[:, :, 0] -= 93.5940
  im[:, :, 1] -= 104.7624
  im[:, :, 2] -= 129.1863
  im = np.expand_dims(im, axis=0)

  res = model.predict(im)
  print(np.argmax(res[0]))