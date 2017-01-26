"""
Created on Jan 26, 2017

Compares faces by embeddings distance

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from keras.preprocessing import image
from skimage import io

from cnn.tpe.cnn_files import training_file
from cnn.tpe.preprocessing import FaceDetector, FaceAligner, clip_to_range
import numpy as np


_files = training_file()
_model_dir = _files.model_dir
_landmarks = _files.join_and_init_path(_model_dir, 'shape_predictor_68_face_landmarks.dat')
_template = _files.join_and_init_path(_model_dir, 'face_template.npy')
fd = FaceDetector()
fa = FaceAligner(_landmarks, _template)

def load_face_from_image(img, imsize=96, border=0):
  """Loads faces from image
    img - binary image
      imsize - image size
      border - image border
    Returns:
      face_tensor - face image tensor
  """
  
  total_size = imsize + 2 * border
  faces = fd.detect_faces(img, get_top=1)
  if len(faces) == 0:
    face_tensor = None
  else:
    face = fa.align_face(img, faces[0], dim=imsize, border=border).reshape(1, total_size, total_size, 3)
    face = clip_to_range(face)
    face_tensor = face.astype(np.float32)
  
  return face_tensor

def load_file(filename, imsize=96, border=0):
  """Generates face image tensor
    Args:
      filename - image file path
      imsize - image size
      border - image border
    Returns:
      face_tensor - face image tensor
  """
    
  img = io.imread(filename)
  #(height, width, _) = img.shape
  print(img.shape)
  face_tensor = load_face_from_image(img, imsize=imsize, norder=border)
  
  return face_tensor

def load_image(filename, border=0):
  """Load faces from images
    Args:
      filename - image file name
      border - image border
    Returns:
      image_tensor - tensor of face image
  """
  
  img = image.load_img(filename, target_size=(160, 160))
  x = image.img_to_array(img)
  arr = np.asarray(x)
  print(img.shape)
  face_tensor = load_face_from_image(arr, imsize=160, norder=border)
  print(face_tensor)
  
  return face_tensor

if __name__ == '__main__':
  """Generates tensors from images"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--filename',
                          type=str,
                          help='Image file name')
  (argument_flags, _) = arg_parser.parse_known_args()
  if argument_flags.filename:
    face_tensor = load_image(argument_flags.filename)
