"""
Created on Jan 26, 2017

Compares faces by embeddings distance

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def load_file(filename, imsize=96, border=0):
  """Generates face image tensor
    Args:
      filename - image file path
      imsize - image size
      border - image border
    Returns:
      face_tensor - face image tensor
  """
    
  total_size = imsize + 2 * border

  img = io.imread(filename)
  faces = fd.detect_faces(img, get_top=1)
  if len(faces) == 0:
    face_tensor = None
  else:
    face = fa.align_face(img, faces[0], dim=imsize, border=border).reshape(1, total_size, total_size, 3)
    face = clip_to_range(face)
    face_tensor = face.astype(np.float32)
  
  return face_tensor
