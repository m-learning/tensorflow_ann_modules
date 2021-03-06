"""
Created on Jan 10, 2017

Face detector on images

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from scipy import misc

from cnn.faces import detect_face
from cnn.faces import facenet
from cnn.faces.cnn_files import training_file
import numpy as np
import tensorflow as tf


def face_detector(img_path, margin, image_size, output_path):
  """Detects face in passed image file
    Args:
      img_path - path to image file
  """
  
  minsize = 20  # minimum size of face
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
  factor = 0.709  # scale factor
  
  with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, _files.model_dir)
  
  try:
      img = misc.imread(img_path)
  except (IOError, ValueError, IndexError) as e:
      errorMessage = '{}: {}'.format(img_path, e)
      print(errorMessage)
  else:
      if img.ndim < 2:
          print('Unable to align "%s"' % img_path)
          return
      if img.ndim == 2:
          img = facenet.to_rgb(img)
      img = img[:, :, 0:3]
  
  bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
  nrof_faces = bounding_boxes.shape[0]
  if nrof_faces > 0:
    det = bounding_boxes[:, 0:4]
    img_size = np.asarray(img.shape)[0:2]
    if nrof_faces > 1:
      bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
      img_center = img_size / 2
      offsets = np.vstack([ (det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0] ])
      offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
      index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
      det = det[index, :]
    det = np.squeeze(det)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    misc.imsave(output_path, scaled)
      
    print(bb)
      
def parse_arguments(argv):
  """Parses and retrieves command line arguments
    Args:
      argv = command line arguments
    Returns:
      parsed and retrieved arguments
  """

  parser = argparse.ArgumentParser()
  
  global _files
  _files = training_file()
  parser.add_argument('--img_path',
                      type=str,
                      help='Image path for recognition.')
  parser.add_argument('--output_path',
                      type=str,
                      help='Output image path.')
  parser.add_argument('--image_size',
                      type=int,
                      default=182,
                      help='Image size (height, width) in pixels.')
  parser.add_argument('--margin',
                      type=int,
                      default=44,
                      help='Margin for the crop around the bounding box (height, width) in pixels.')
  parser.add_argument('--random_order',
                      help='Shuffles the order of images to enable alignment using multiple processes.',
                      action='store_true')
  parser.add_argument('--gpu_memory_fraction',
                      type=float,
                      default=1.0,
                      help='Upper bound on the amount of GPU memory that will be used by the process.')
  (argument_flags, _) = parser.parse_known_args()
  
  return argument_flags
  
def parse_and_run(argv):
  """Parses command line arguments and runs face finder
    Args:
      argv - command line arguments array
  """
  
  args = parse_arguments(argv)
  face_detector(args.img_path, args.margin, args.image_size, args.output_path)

if __name__ == '__main__':
  parse_and_run(sys.argv)
