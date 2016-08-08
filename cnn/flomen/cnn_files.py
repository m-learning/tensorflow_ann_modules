'''
Created on Jun 21, 2016

Files for training data

@author: Levan Tsinadze
'''

import os
import sys
import tarfile
import zipfile
import shutil
import glob

from six.moves import urllib
try:
  from PIL import Image
except ImportError:
  print "Importing Image from PIL threw exception"
  import Image
#import Image

from cnn.utils.file_utils import cnn_file_utils

# Files and directory constant parameters
TRAINIG_SET_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
PERSONS_SETS = ['http://www.emt.tugraz.at/~pinz/data/GRAZ_01/persons.zip',
                'http://www.emt.tugraz.at/~pinz/data/GRAZ_02/person.zip',
                'http://www.emt.tugraz.at/~pinz/data/GRAZ_01/bikes.zip',
                'http://www.emt.tugraz.at/~pinz/data/GRAZ_02/cars.zip',
                'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',
                'http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip']
TRAINIG_ZIP_FOLDER = 'training_arch'
CASSIFICATION_DIRS = ['persons', 'persons', 'bikes', 'cars', 'persons', 'persons']

# Directories to move training data from
PERSON_DIR = 'person'
PEDESTRIAN_DIR = 'PennFudanPed'
PEDESTRIAN_IMG_DIR = 'PNGImages'
PERSONS_JPEG_DIR = 'JPEGImages'

# Files and directories for parameters (trained), training, validation and test
class training_file(cnn_file_utils):
  
  def __init__(self):
    super(training_file, self).__init__('flomen')
    
  # Converts person images
  def convert_person_images(self, prfx, src_dir, persons_dir, img_type):
    
    i = 0
    scan_persons_dir = os.path.join(src_dir, img_type)
    for pr in glob.glob(scan_persons_dir):
      im = Image.open(pr)
      fl_name = prfx + 'cnvrt_prs_' + str(i) + '.jpg'
      n_im = os.path.join(persons_dir, fl_name)
      if not os.path.exists(n_im):
        im.save(n_im)
        os.remove(pr)
      i += 1
  
  # Gets persons training set
  def get_persons_dir(self, dest_directory, zip_ref):
    
    pers_dir = os.path.join(dest_directory , PERSON_DIR)
    
    if os.path.exists(pers_dir):
      shutil.rmtree(pers_dir, ignore_errors=True)
    zip_ref.extractall(dest_directory)
    
    return pers_dir
  
  # Gets persons training set
  def get_pedestrians_dir(self, dest_directory, zip_ref):
    
    extr_dir = os.path.join(dest_directory , PEDESTRIAN_DIR)
    pers_dir = os.path.join(extr_dir , PEDESTRIAN_IMG_DIR)
    if os.path.exists(pers_dir):
      shutil.rmtree(extr_dir, ignore_errors=True)
    zip_ref.extractall(dest_directory)
    
    return pers_dir
  
  # Gets persons JPEG files training set
  def get_persons_jpeg_dir(self, dest_directory, zip_ref):
    
    pers_dir = os.path.join(dest_directory , PERSONS_JPEG_DIR)
    if os.path.exists(pers_dir):
      shutil.rmtree(pers_dir, ignore_errors=True)
    zip_ref.extractall(dest_directory)
  
  # Gets persons dat aset
  def get_persons_set(self, dest_directory):
    
    training_dir = self.get_data_directory()
    for i in  range(len(PERSONS_SETS)):
      prfx = str(i) + '_'
      person_set = PERSONS_SETS[i]
      filename = prfx + person_set.split('/')[-1]
      filepath = os.path.join(dest_directory, filename)
      if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(person_set, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
      zip_ref = zipfile.ZipFile(filepath, 'r')
      persons_dir = os.path.join(training_dir , CASSIFICATION_DIRS[i])
      img_type = '*.bmp'
      if i == 1:
        pers_dir = self.get_persons_dir(dest_directory, zip_ref)
      elif i == 4:
        img_type = '*.png'
        pers_dir = self.get_pedestrians_dir(dest_directory, zip_ref)
      elif i == 5:
        img_type = '*.jpg'
        pers_dir = self.get_persons_jpeg_dir(dest_directory, zip_ref)
      else:
        zip_ref.extractall(training_dir)
        pers_dir = persons_dir
      self.convert_person_images(prfx, pers_dir, persons_dir, img_type)      
    
  
  # Gets or generates training set
  def get_or_init_training_set(self):
    
    dest_directory = self.join_path(self.get_data_general_directory, TRAINIG_ZIP_FOLDER)
    if not os.path.exists(dest_directory):
      os.mkdir(dest_directory)  
      
    filename = TRAINIG_SET_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(TRAINIG_SET_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    training_dir = self.get_training_directory()
    if not os.path.exists(training_dir):
      os.mkdir(training_dir)  
    else:
      shutil.rmtree(training_dir, ignore_errors=True)
      os.mkdir(training_dir)
    tarfile.open(filepath, 'r:gz').extractall(training_dir)
    self.get_persons_set(dest_directory)
    print 'Training set is prepared'
