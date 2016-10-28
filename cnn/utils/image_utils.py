"""
Created on Oct 19, 2016
Utility class for images
@author: levan-lev
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import imghdr
import os
import sys
import traceback

from cnn.utils.pillow_resizing import pillow_resizer


try:
  from PIL import Image
except ImportError:
  print("Importing Image from PIL threw exception")
  import Image

resize_image = False
verbose_error = None

IMAGE_RGB_FORMAT = 'RGB'
IMAGE_SAVE_FORMAT = 'jpeg'

IMAGE_SIZE = 299

class image_converter(object):
  """Utility class for image manipulation"""
  
  def __init__(self, from_parent, to_dir, prefx):
    self.from_parent = from_parent
    self.to_dir = to_dir
    self.prefx = prefx
    self.resizer = pillow_resizer(IMAGE_SIZE)
    
  def convert_image(self, im):
    """Converts PNG images to JPG format
      Args:
        im - image
      Return:
        jpg_im - converted image
    """
    jpg_im = im.convert(IMAGE_RGB_FORMAT)
    return jpg_im
    
  
  def write_file(self, pr, i, jpg_im=None):
    """Converts and writes file
      Args:
        pr - path for source file
        i - index for file name suffix
    """
    fl_name = self.prefx + '_' + 'cnvrt_data_' + str(i) + '.jpg'
    n_im = os.path.join(self.to_dir, fl_name)
    if os.path.exists(n_im):
      os.remove(n_im)
    if jpg_im is None:
      im = Image.open(pr)
    else:
      im = jpg_im
    print(im)
    im.save(n_im)
    
  def resize_if_nedded(self, im):
    """Resizes passed image if configured
      Args:
        im - image
      Returns:
        img - resized image
    """
    
    if resize_image:
      img = self.resizer.resize_thumbnail(im)
    else:
      img = im
    
    return img
    
  def write_file_quietly(self, pr, i):
    """Converts and writes file and logs errors
      Args:
        pr - path for source file
        i - index for file name suffix    
    """
    try:
      file_type = imghdr.what(pr)
      im = Image.open(pr)
      if file_type in ('jpg:', 'jpeg', 'JPG:', 'JPEG'):
        img = self.resize_if_nedded(im)
        self.write_file(pr, i, img)
      elif file_type in ('png', 'PNG'):
        jpg_im = self.convert_image(im)
        img = self.resize_if_nedded(jpg_im)
        self.write_file(pr, i, img)
        print("Image is converted" , pr , "\n")
      else:
        print('incorrect file type - ', file_type)
    except IOError:
      print('Error for - ', pr)
      if verbose_error:
        traceback.print_exc()
      else:
        os.remove(pr)   
    
  def migrate_images(self):
    """Converts and migrates images from one 
      directory to other"""
    i = 0
    from_dirs = os.listdir(self.from_parent)
    for from_dir in from_dirs:
      scan_dir = os.path.join(self.from_parent, from_dir, '*.jpg')
      print(scan_dir)
      for pr in glob.glob(scan_dir):
        self.write_file_quietly(pr, i)
        i += 1
        
if __name__ == '__main__':
  
  call_args = sys.argv
  from_dirs = call_args[1]
  to_dir = call_args[2]
  prefx = call_args[3]
  
  if len(call_args) > 4:
    resize_image = True
  
  if len(call_args) > 5:
    verbose_error = True
    
  from_dirs_list = os.listdir(from_dirs)
  print(from_dirs)
  print(from_dirs_list)
  print(to_dir)
  print(prefx)
  
  converter = image_converter(from_dirs, to_dir, prefx)
  converter.migrate_images()
