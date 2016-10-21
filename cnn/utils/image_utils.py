'''
Created on Oct 19, 2016
Utility class for images
@author: levan-lev
'''

import glob
import imghdr
import os
import sys
import traceback


try:
  from PIL import Image
except ImportError:
  print "Importing Image from PIL threw exception"
  import Image
  
verbose_error = None

class image_converter(object):
  """Utility class for image manipulation"""
  
  def __init__(self, from_parent, to_dir, prefx):
    self.from_parent = from_parent
    self.to_dir = to_dir
    self.prefx = prefx
  
  def write_file(self, pr, i):
    """Converts and writes file
      Args:
        pr - path for source file
        i - index for file name suffix
    """
    fl_name = self.prefx + '_' + 'cnvrt_data_' + str(i) + '.jpg'
    n_im = os.path.join(self.to_dir, fl_name)
    if os.path.exists(n_im):
      os.remove(n_im)
    im = Image.open(pr)
    print im
    im.save(n_im)
    
  def write_file_quietly(self, pr, i):
    """Converts and writes file and logs errors
      Args:
        pr - path for source file
        i - index for file name suffix    
    """
    try:
      file_type = imghdr.what(pr)
      if file_type in ('jpg:', 'jpeg'):
        self.write_file(pr, i)
      else:
        print('incorrect file type - ', file_type)
    except IOError:
      print('Error for - ', pr)
      if verbose_error:
        traceback.print_exc()
      else:
        os.remove(pr)   
    
  def migrate_images(self):
    """Converts and migrates images from one directory to other
    """
    i = 0
    from_dirs = os.listdir(self.from_parent)
    for from_dir in from_dirs:
      scan_dir = os.path.join(self.from_parent, from_dir, '*.jpg')
      print scan_dir
      for pr in glob.glob(scan_dir):
        self.write_file_quietly(pr, i)
        i += 1
        
if __name__ == '__main__':
  
  call_args = sys.argv
  from_dirs = call_args[1]
  to_dir = call_args[2]
  prefx = call_args[3]
  
  if len(call_args) > 4:
    verbose_error = True
  
  from_dirs_list = os.listdir(from_dirs)
  print from_dirs
  print from_dirs_list
  print to_dir
  print prefx
  
  converter = image_converter(from_dirs, to_dir, prefx)
  converter.migrate_images()
