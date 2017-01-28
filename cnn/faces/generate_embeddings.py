"""
Created on Jan 18, 2017

Generates face embeddings from images

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.faces import face_utils as utils
from cnn.faces import network_interface as interface


def generate_face_embeddings(image_parameters):
  """Generates many face embeddings from files and calculates L2 distances
    Args:
      args - command line arguments
    Returns:
      face_embeddings - collection of
        image file name and embedding vectors
  """
  
  face_embeddings = []
  
  (image_files, model_dir, image_size, margin, gpu_memory_fraction, _files) = image_parameters
  images = utils.load_and_align_data(image_files, image_size, margin, gpu_memory_fraction, _files)
  emb = interface.calculate_embeddings(model_dir, images)
          
  nrof_images = len(image_files)

  print('Images:')
  for i in range(nrof_images):
    print('%1d: %s' % (i, image_files[i]))
  print('')
  
  # Print distance matrix
  print('Distance matrix')
  print('    ', end='')
  
  for i in range(nrof_images):
    face_embedding = (image_files[i], emb[i, :])
    face_embeddings.append(face_embedding)
  
  return face_embeddings
