"""
Created on Jan 26, 2017

Compares faces by embeddings distance

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from skimage import io

from cnn.tpe.network_model import FaceVerificator


# ##
dist = 0.85
# ##

def _compare_faces(iamge1, image2):
  """Compares two faces
    Returns:
      tuple of -
        scores - compared scores
        comps - compared results
  """
  fv = FaceVerificator('./model')
  fv.initialize_model()
  
  img_0 = io.imread(iamge1)
  img_1 = io.imread(image2)
  
  faces_0 = fv.process_image(img_0)
  faces_1 = fv.process_image(img_1)
  
  n_faces_0 = len(faces_0)
  n_faces_1 = len(faces_1)
  
  if n_faces_0 == 0 or n_faces_1 == 0:
      print('Error: No faces found on the {}!'.format(iamge1 if n_faces_0 == 0 else image2))
      exit()
  
  rects_0 = list(map(lambda p: p[0], faces_0))
  rects_1 = list(map(lambda p: p[0], faces_1))
  
  embs_0 = list(map(lambda p: p[1], faces_0))
  embs_1 = list(map(lambda p: p[1], faces_1))
  
  (scores, comps) = fv.compare_many(dist, embs_0, embs_1)

  print('Rects on image 0: {}'.format(rects_0))
  print('Rects on image 1: {}'.format(rects_1))

  # print('Embeddings of faces on image 0:')
  # print(embs_0)
  #
  # print('Embeddings of faces on image 1:')
  # print(embs_1)
  
  print('Score matrix:')
  print(scores)
  
  print('Decision matrix :')
  print(comps)
  
  return (scores, comps)

if __name__ == '__main__':
  """Generates tensors from images"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--image1',
                          type=str,
                          help='First image file name')
  arg_parser.add_argument('--image2',
                          type=str,
                          help='Second image file name')
  (flags, _) = arg_parser.parse_known_args()
  if flags.image1 and flags.image2:
    comp_result = _compare_faces(flags.image1, flags.image2)
  else:
    print('No images to compare')
