import os

from cnn.utils.file_utils import cnn_file_utils
import numpy as np


PARENT_DIR = 'fcn'

PATH_FOR_VGG = 'vgg16.npy'
PATH_FOR_TEST = 'test_data'

def color_image(image, num_classes=20):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    
    return mycm(norm(image))
  
  
def get_wgg_weights():
  
  cnn_files = cnn_file_utils(PARENT_DIR)
  vgg_path = cnn_files.init_files_directory();
  vgg_weights = cnn_files.join_path(vgg_path, PATH_FOR_VGG)
  
  return vgg_weights

def get_test_dir():
  
  cnn_files = cnn_file_utils(PARENT_DIR)
  test_dir = cnn_files.join_path(cnn_files.get_data_general_directory, PATH_FOR_TEST);
  
  return test_dir

def get_test_results_dir():
  
  test_dir = get_test_dir()
  test_result_dir = os.path.join(test_dir, 'results')
  
  return test_result_dir
