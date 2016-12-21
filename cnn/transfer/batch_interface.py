"""
Created on Dec 13, 2016

Runs retrained neural network interface on batch of images

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import io
import os
import shutil

from cnn.transfer.recognizer_interface import retrained_recognizer
from cnn.utils import file_utils as files
from cnn.utils import image_color_refiner as refiner
from cnn.utils.pillow_resizing import pillow_resizer
import tensorflow as tf


try:
  from PIL import Image
except ImportError:
  print("Importing Image from PIL threw exception")
  import Image

IMAGE_SAVE_FORMAT = 'jpeg'

resizer = pillow_resizer(299)

class batch_recognizer(retrained_recognizer):
  """Class to run recognition by trained network on batch of files"""
  
  def __init__(self, argument_flags, training_file_const=None):
    super(batch_recognizer, self).__init__(training_file_const)
    self.__argument_flags = argument_flags
    
  def crop_image(self, im):
    """Crops passed image if required
      Args:
        im - image to crop
      Returns:
        cropped_im - cropped image
    """
    
    if self.__argument_flags.box_images:
      [x, y] = im.size
      left = (x - x / 10) / 2
      top = (y - y / 20) / 2
      right = (x + x / 1.5) / 2
      bottom = (y + y / 4) / 2
      box = [left, top, right, bottom]
      cropped_im = im.crop(box)
      im.close()
    else:
      cropped_im = im
      
    return cropped_im
  
  def resize_image(self, im):
    """Resizes image for recognition
      Args:
        im - binary image
      Returns:
        mdf - resized image
    """
    
    cropped_im = self.crop_image(im)
    img = resizer.resize_full(cropped_im)
    # mdf = color.sharpen_edges(img)
    mdf = img
    refiner.color_refinement(mdf)
    
    return mdf
  
  def to_byte_array(self, jpg_im):
    """Converts image to byte Array
      Args:
        jpg_im - image
      Return:
        img_array - byte array of image
    """
    
    img_bytes = io.BytesIO()
    jpg_im.save(img_bytes, format=IMAGE_SAVE_FORMAT)
    im_arr = img_bytes.getvalue()
    
    return im_arr
  
  def resize_and_binarize(self, image_path):
    """Resizes image and recognizes
      Args:
        image_path - image file path
      Returns:
        answer - prediction result
    """
    
    image_data = Image.open(image_path)
    img = self.resize_image(image_data)
    im_arr = self.to_byte_array(img)
    
    return im_arr
    
  def recognize_batch(self):
    """Runs recognition of images"""
    
    model_dir = argument_flags.model_path
    model_path = os.path.join(model_dir, files.WEIGHTS_FILE)
    label_path = os.path.join(model_dir, files.LABELS_FILE)
    dst_dir = self.__argument_flags.dst_dir
    if not os.path.exists(dst_dir):
      os.mkdir(dst_dir)
    scan_dir = os.path.join(self.__argument_flags.src_dir, '*.jpg')
    with tf.Session() as sess:
      for pr in glob.glob(scan_dir):
        image_data = self.resize_and_binarize(pr)
        answer = self.recognize_by_data(sess, image_data, model_path, label_path)
        if answer is not None and len(answer) > 0:
          print(pr, answer)
          folder_path = os.path.join(dst_dir, answer)
          if not os.path.exists(folder_path):
            os.mkdir(folder_path)
          shutil.copy2(pr, folder_path)
      
    
def init_parameters_and_run(argument_flags):
  """Initializes parameters and runs recognition
    Args:
      argument_flags - argument parsers
  """
  
  if argument_flags.src_dir and argument_flags.dst_dir and argument_flags.model_path:
    recognizer = batch_recognizer(argument_flags)
    recognizer.recognize_batch()
  
if __name__ == '__main__':
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--model_path',
                          type=str,
                          help='Model directory')
  arg_parser.add_argument('--src_dir',
                          type=str,
                          help='Source directory')
  arg_parser.add_argument('--dst_dir',
                          type=str,
                          help='Destination directory')
  arg_parser.add_argument('--box_images',
                          dest='box_images',
                          action='store_true',
                          help='Crop images.')
  arg_parser.add_argument('--not_box_images',
                          dest='box_images',
                          action='store_false',
                          help='Do not Crop images.')
  (argument_flags, _) = arg_parser.parse_known_args()
  init_parameters_and_run(argument_flags)
  