"""
Created on Nov 25, 2016
Utility class for image processing
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.utils.image_utils import image_converter


IMAGE_SIZE = 299

class document_image_converter(image_converter):
  """Image converter implementation with cropping"""
  
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
    super(document_image_converter, self).__init__(from_parent, to_dir, prefx,
                                                   rotate_image=rotate_image,
                                                   rotate_pos=rotate_pos,
                                                   add_extensions=add_extensions,
                                                   image_extensions=image_extensions,
                                                   change_extensions=change_extensions,
                                                   resize_image=resize_image,
                                                   box_image=box_image,
                                                   box_sizes=box_sizes,
                                                   image_size=image_size)
    
  def crop_image(self, im):
    """Document image croping"""
    
    [x, y] = im.size
    left = (x - x / 5) / 2
    top = (y - y / 3.5) / 2
    right = x
    bottom = (y + y / 2.6) / 2
    box = [left, top, right, bottom]
    croped_image = im.crop(box)
    
    return croped_image    
    
  def resize_if_nedded(self, im):
    """Resizes passed image if configured
      Args:
        im - image
      Returns:
        img - resized image
    """
    
    if self.resize_image:
      if self.box_image:
        [x, y] = im.size
        left = (x - x / 5) / 2
        top = (y - y / 3.5) / 2
        right = x
        bottom = (y + y / 2.6) / 2
        box = [left, top, right, bottom]
        im_to_write = im.crop(box)
      else:
        im_to_write = im
      img = self.resizer.resize_thumbnail(im_to_write)
    else:
      img = im
    
    return img    

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
  
  if argument_flags.box_images and argument_flags.box_sizes:
    box_image = True
    sizes = argument_flags.box_sizes.split('-')
    box_sizes = [int(sizes[0]), int(sizes[1])]
  else:
    box_image = False
    box_sizes = []
  
  converter = document_image_converter(from_dirs, to_dir, prefx,
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
