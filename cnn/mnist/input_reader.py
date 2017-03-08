"""
Created on Jun 17, 2016
Image preparation for network interface
@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import math
from scipy import ndimage

import numpy as np


# Image sizes
IMAGE_SIZE = 28

# Input for fully connected layer
n_input = 784  # MNIST data input (img shape: 28*28)

def getBestShift(img):
  """Gets best shift for image
    Args:
      img - image
    Returns:
      shiftx - shift by x
      shifty - shift by y
  """
    
  cy, cx = ndimage.measurements.center_of_mass(img)

  rows, cols = img.shape
  shiftx = np.round(cols / 2.0 - cx).astype(int)
  shifty = np.round(rows / 2.0 - cy).astype(int)

  return shiftx, shifty

def shift(img, sx, sy):
  """Shifts image
    Args:
      img - image
      sx - x shift
      sy - y shift
    Returns:
      shifted - modified (shifted) image
  """
  
  rows, cols = img.shape
  M = np.float32([[1, 0, sx], [0, 1, sy]])
  shifted = cv2.warpAffine(img, M, (cols, rows))
  
  return shifted

def read_input_file(image_file_path):
  """Reads input file for recognition
    Args:
      image_file_path - path to image
    Returns:
      image_rec - image tensor
  """
    
  image_rec = np.zeros((1, n_input))
  gray = cv2.imread(image_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  # rescale it
  gray = cv2.resize(255 - gray, (28, 28))
  # better black and white version
  (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  print(thresh)

  while np.sum(gray[0]) == 0:
      gray = gray[1:]

  while np.sum(gray[:, 0]) == 0:
      gray = np.delete(gray, 0, 1)

  while np.sum(gray[-1]) == 0:
      gray = gray[:-1]

  while np.sum(gray[:, -1]) == 0:
      gray = np.delete(gray, -1, 1)

  rows, cols = gray.shape

  if rows > cols:
      factor = 20.0 / rows
      rows = 20
      cols = int(round(cols * factor))
      # first cols than rows
      gray = cv2.resize(gray, (cols, rows))
  else:
      factor = 20.0 / cols
      cols = 20
      rows = int(round(rows * factor))
      # first cols than rows
      gray = cv2.resize(gray, (cols, rows))

  colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
  rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
  gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

  shiftx, shifty = getBestShift(gray)
  shifted = shift(gray, shiftx, shifty)
  gray = shifted
  
  image_modified_file = image_file_path + '_modf.png'
  # write_image
  cv2.imwrite(image_modified_file, gray)
  
  flatten = gray.flatten() / 255.0
  image_rec[0] = flatten

  return image_rec
