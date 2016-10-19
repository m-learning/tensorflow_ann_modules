'''
Created on Oct 19, 2016
Utility class for images
@author: levan-lev
'''

import glob
import os
import sys

class image_converter(object):
  """
    Utility class for image manipulation
  """
  
  def __init__(self, from_parent, to_dir, prefx):
    self.from_parent = from_parent
    self.to_dir = to_dir
    self.prefx = prefx
    self.image_resizer
    
  def migrate_images(self):
    """
      Converts and migrates images from one directory to other
    """
    i = 0
    
    from_dirs = os.listdir(self.from_parent)
    for from_dir in from_dirs:
      scan_dir = os.path.join(from_dir, 'jpg')
      for pr in glob.glob(scan_dir):
        fl_name = self.prefx + 'cnvrt_data_' + str(i) + '.jpg'
        n_im = os.path.join(self.to_dir, fl_name)
        if os.path.exists(n_im):
          os.remove(n_im)
        self.image_resizer.read_resize_write(pr, n_im)
        i += 1
        
if __name__ == '__main__':
  
  call_args = sys.argv
  from_dirs_txt = call_args[1]
  to_dir = call_args[2]
  prefx = call_args[3]
  
  from_dirs = from_dirs_txt.split(',')
  
  converter = image_converter(from_dirs, to_dir, prefx)
  converter.migrate_images()