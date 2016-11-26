"""
Created on Oct 19, 2016
Utility class for images
@author: levan-lev
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import imghdr
import os
import traceback

from cnn.utils import file_utils
from cnn.utils.pillow_resizing import pillow_resizer


try:
  from PIL import Image
except ImportError:
  print("Importing Image from PIL threw exception")
  import Image

verbose_error = None

IMAGE_RGB_FORMAT = 'RGB'
IMAGE_SAVE_FORMAT = 'jpeg'

IMAGE_SIZE = 299

class image_indexer(object):
  """Image parameters for indexing"""
  
  def __init__(self, rotate_angles):
    self._rotate_angles = rotate_angles
    self._index = 0
    
  def incr_indexer(self):
    """Increments image index"""
    self._index += 1
  
  @property
  def curr_index(self):
    return self._index
  
  @property
  def rotate_angles(self):
    return self._rotate_angles

class image_converter(object):
  """Utility class for image manipulation"""
  
  def __init__(self, from_parent, to_dir, prefx,
               rotate_image=True,
               rotate_pos=[],
               add_extensions=True,
               image_extensions=['jpg', 'png'],
               change_extensions=['gif'],
               resize_image=True,
               box_image=False,
               box_sizes=[],
               image_size=IMAGE_SIZE):
    self.from_parent = from_parent
    self.to_dir = to_dir
    self.prefx = prefx
    self.resizer = pillow_resizer(IMAGE_SIZE)
    self.rotate_image = rotate_image
    self.rotate_pos = rotate_pos
    self.add_extensions = add_extensions
    self.image_extensions = image_extensions
    self.change_extensions = change_extensions
    self.resize_image = resize_image
    self.image_size = image_size
    self.box_image = box_image
    self.box_sizes = box_sizes
    self.indexer = image_indexer(self.rotate_pos)
    self.converted_paths = set()
    
  def convert_image(self, im):
    """Converts PNG images to JPG format
      Args:
        im - image
      Return:
        jpg_im - converted image
    """
    jpg_im = im.convert(IMAGE_RGB_FORMAT)
    return jpg_im
    
  
  def write_file(self, pr, jpg_im=None):
    """Converts and writes file
      Args:
        pr - path for source file
        i - index for file name suffix
      Returns:
        im - saved image
    """
    fl_name = self.prefx + '_' + 'cnvrt_data_' + str(self.indexer.curr_index) + '.jpg'
    file_utils.ensure_dir_exists(self.to_dir)
    n_im = os.path.join(self.to_dir, fl_name)
    if os.path.exists(n_im):
      os.remove(n_im)
    if jpg_im is None:
      im = Image.open(pr)
    else:
      im = jpg_im
    print(pr, im, n_im)
    im.save(n_im)
    
    return im
  
  def crop_image(self, im):
    """Validates if flag set crops passed image
      Args:
        im - image to crop
      Return:
        cropped_image - cropped image
    """
    
    if self.box_image:
      [x, y] = im.size
      left = (x - x / 5) / 2
      top = (y - y / 3.5) / 2
      right = x
      bottom = (y + y / 2.6) / 2
      box = [left, top, right, bottom]
      cropped_image = im.crop(box)
    else:
      cropped_image = im
      
    return cropped_image
    
  def resize_if_nedded(self, im):
    """Resizes passed image if configured
      Args:
        im - image
      Returns:
        img - resized image
    """
    
    im_to_write = self.crop_image(im)
    if self.resize_image:
      img = self.resizer.resize_full(im_to_write)
    else:
      img = im
    
    return img
    
  def write_file_quietly(self, pr):
    """Converts and writes file and logs errors
      Args:
        pr - path for source file
        i - index for file name suffix 
      Returns:
        saved_im - saved image
    """
    
    try:
      file_type = imghdr.what(pr)
      im = Image.open(pr)
      if file_type in ('jpg:', 'jpeg', 'JPG:', 'JPEG'):
        img = self.resize_if_nedded(im)
        saved_im = self.write_file(pr, img)
      elif file_type in ('png', 'PNG'):
        jpg_im = self.convert_image(im)
        img = self.resize_if_nedded(jpg_im)
        saved_im = self.write_file(pr, img)
        print("Image is converted" , pr , "\n")
      else:
        print('incorrect file type - ', file_type)
        saved_im = None
    except IOError:
      print('Error for - ', pr)
      saved_im = None
      if verbose_error:
        traceback.print_exc()
      else:
        os.remove(pr)   
        
    return saved_im
  
  def resize_and_write(self, pr, im):
    """Resizes and saves image
      Args:
        pr - source image
        i - index for file name suffix
        im - image to resize and save
    """
    img = self.resize_if_nedded(im)
    self.write_file(pr, img)
    
  def rotate_and_write(self, rotate_params):
    """Rotates and writes images in destination directory
      Args:
        rotate_params - rotation parameters
    """
    
    (pr, im) = rotate_params
    
    rotate_angles = self.indexer.rotate_angles
    if im is not None: 
      if self.rotate_image and rotate_angles is not None:
        for ang in rotate_angles:
          if type(ang) is str:
            ang_int = int(ang)
          else:
            ang_int = ang
          im_r = im.rotate(ang_int, expand=True)
          self.resize_and_write(pr, im_r)
          self.indexer.incr_indexer()
      else:
        self.resize_and_write(pr, im)
        self.indexer.incr_indexer()
        
  def add_extension_to_images(self, raw_path, parent_dir_path=None):
    """Adds extensions to files
      Args:
        dir_path - directory to scan
    """
    
    if os.path.isdir(raw_path):
      file_names = os.listdir(raw_path)
      for file_name in file_names:
        self.add_extension_to_images(file_name, parent_dir_path=raw_path)
    else:
      full_file_path = os.path.join(parent_dir_path, raw_path)
      ext = os.path.splitext(full_file_path)[-1].lower()
      if not ext or (self.change_extensions is not None and ext in self.change_extensions):
        file_name_with_ext = full_file_path + '.jpg'
        os.rename(full_file_path, file_name_with_ext)
    
    
  def migrate_images(self):
    """Converts and migrates images from one 
       directory to other"""
    
    from_dirs = os.listdir(self.from_parent)
    for from_dir in from_dirs:
      if self.add_extensions:
        full_file_path = os.path.join(self.from_parent, from_dir)
        self.add_extension_to_images(full_file_path)
      for ext in self.image_extensions:
        scan_ext = '*.' + ext
        scan_dir = os.path.join(self.from_parent, from_dir, scan_ext)
        print(scan_dir)
        for pr in glob.glob(scan_dir):
          if pr not in self.converted_paths:
            im = self.write_file_quietly(pr)
            self.indexer.incr_indexer()
            rotate_params = (pr, im)
            if self.rotate_image:
              self.rotate_and_write(rotate_params)
          self.converted_paths.add(pr)
          
        

def run_image_processing(argument_flags):
  """Runs image processing
    Args:
      argument_flags - Command line arguments
  """
  
  from_dirs = argument_flags.src_dir
  to_dir = argument_flags.dst_dir
  prefx = argument_flags.file_prefix
  rotate_image = argument_flags.rotate_images
  if argument_flags.rotate_pos:
    rotate_pos = argument_flags.rotate_pos.split('|')
  else:
    rotate_pos = []
    
  add_extensions = argument_flags.add_extensions
  
  resize_image = argument_flags.resize_images
  image_size = argument_flags.image_size
      
  image_extensions = argument_flags.image_extensions.split('-')
  change_extensions = argument_flags.change_ext.split('-')
  
  if argument_flags.log_errors:
    verbose_error = argument_flags.log_errors
  
  if argument_flags.box_images and argument_flags.box_sizes:
    box_image = True
    sizes = argument_flags.box_sizes.split('-')
    box_sizes = [int(sizes[0]), int(sizes[1])]
  else:
    box_image = False
    box_sizes = []
  
  converter = image_converter(from_dirs, to_dir, prefx,
                              rotate_image=rotate_image,
                              rotate_pos=rotate_pos,
                              add_extensions=add_extensions,
                              image_extensions=image_extensions,
                              change_extensions=change_extensions,
                              resize_image=resize_image,
                              box_image=box_image,
                              box_sizes=box_sizes,
                              image_size=image_size)
  converter.migrate_images()
      

def read_arguments_and_run():
  """Retrieves command line arguments for image processing"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--src_dir',
                          type=str,
                          help='Source directory.')
  arg_parser.add_argument('--dst_dir',
                          type=str,
                          help='Destination directory.') 
  arg_parser.add_argument('--file_prefix',
                          type=str,
                          default='trn_',
                          help='Converted file prefix.') 
  arg_parser.add_argument('--resize_images',
                          type=bool,
                          default=True,
                          help='Resize images.')
  arg_parser.add_argument('--box_images',
                          dest='box_images',
                          action='store_true',
                          help='Crop images.')
  arg_parser.add_argument('--not_box_images',
                          dest='box_images',
                          action='store_false',
                          help='Do not Crop images.')
  arg_parser.add_argument('--box_sizes',
                          type=str,
                          default='2-2',
                          help='Resize images.')
  
  arg_parser.add_argument('--image_size',
                          type=int,
                          default=IMAGE_SIZE,
                          help='Images Size.')
  
  arg_parser.add_argument('--log_errors',
                          type=bool,
                          default=False,
                          help='Log errors.')
  
  arg_parser.add_argument('--add_extensions',
                          type=bool,
                          default=True,
                          help='Adds extensions to images.')
  
  arg_parser.add_argument('--image_extensions',
                          type=str,
                          default='jpg-png',
                          help='Image extensions to filter.')
  
  arg_parser.add_argument('--rotate_images',
                          dest='rotate_images',
                          action='store_true',
                          help='Rotate images flag')
  
  arg_parser.add_argument('--not_rotate_images',
                          dest='rotate_images',
                          action='store_false',
                          help='Rotate images flag')   
  
  
  arg_parser.add_argument('--rotate_pos',
                          type=str,
                          default='30|-30|45|-45|180|90|-90|95|-95',
                          help='Rotate images (--rotate_pos 30|-30|45|-45).')
  
  arg_parser.add_argument('--change_ext',
                          type=str,
                          default='gif-GIF',
                          help='Change extension to JPG flag')
  (argument_flags, _) = arg_parser.parse_known_args()
  run_image_processing(argument_flags)

        
if __name__ == '__main__':
  """Converts images for training data set"""
  
  read_arguments_and_run()
