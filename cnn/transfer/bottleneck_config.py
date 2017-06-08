"""
Created on Jul 15, 2016

Configures bottleneck cache for training

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import traceback

from tensorflow.python.platform import gfile

from cnn.transfer.dataset_config import MAX_NUM_IMAGES_PER_CLASS
import cnn.transfer.dataset_config as dataset 
from cnn.utils import cnn_flags_utils as conts
from cnn.utils import file_utils
import numpy as np
import tensorflow as tf


def run_bottleneck_on_image(bottleneck_params):
  """Runs inference on an image to extract the 'bottleneck' summary layer.
  Args:
    bottleneck_params - tuple containing
      sess: Current active TensorFlow Session.
      image_data: Numpy array of image data.
      image_data_tensor: Input data layer in the graph.
      bottleneck_tensor: Layer before the final softmax.
  Returns:
    bottleneck_values - numpy array of bottleneck values.
  """
  (sess, image_data, image_data_tensor, bottleneck_tensor) = bottleneck_params
  bottleneck_values = sess.run(bottleneck_tensor,
                              {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  
  return bottleneck_values

def get_or_create_bottleneck_and_path(create_params):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    create_params - tuple to create bottleneck contains
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string  of the subfolders containing the training
      images.
      category: Name string of which  set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      bottleneck_tensor: The output tensor for the bottleneck values.
  Returns:
    String - bottleneck path
    Numpy array of values produced by the bottleneck layer for the image.
  """
  (sess, image_lists, label_name, index, image_dir, category, bottleneck_dir,
   jpeg_data_tensor, bottleneck_tensor, image) = create_params

  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  file_utils.ensure_dir_exists(sub_dir_path)
  if image is None:
    bot_path_params = (image_lists, label_name, index, bottleneck_dir, category)
    bottleneck_path = dataset.get_bottleneck_path(bot_path_params)
  else:
    bottleneck_path = os.path.join(bottleneck_dir, sub_dir, image) + '.txt'
  if not os.path.exists(bottleneck_path):
    print('Creating bottleneck at ', bottleneck_path)
    if image is None:
      img_path_params = (image_lists, label_name, index, image_dir, category)
      image_path = dataset.get_image_path(img_path_params)
    else:
      image_path = os.path.join(image_dir, sub_dir, image)
    try:
      if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
      else:
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_params = (sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_values = run_bottleneck_on_image(bottleneck_params)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
          bottleneck_file.write(bottleneck_string)

      with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
      bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except:
      print('Not valid image ', image_path)
      traceback.print_exc()
      file_utils.remove_file(image_path)
      (bottleneck_values, bottleneck_path) = (None, None)
  
  return (bottleneck_values, bottleneck_path)
  

def get_or_create_bottleneck(create_params):
  """Retrieves or calculates bottleneck values for an image.
  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.
  Args:
    create_params - tuple to create bottleneck contains
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string  of the subfolders containing the training
      images.
      category: Name string of which  set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      bottleneck_tensor: The output tensor for the bottleneck values.
  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
  (bottleneck_values, _) = get_or_create_bottleneck_and_path(create_params)
  return bottleneck_values

def cache_bottlenecks(cache_params):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    cache_params - tuple of -
      scache_paramsess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      image_dir: Root folder string of the subfolders containing the training
      images.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      bottleneck_tensor: The penultimate output layer of the graph.
  """
  (sess, image_lists, image_dir, bottleneck_dir,
   jpeg_data_tensor, bottleneck_tensor) = cache_params
  how_many_bottlenecks = 0
  file_utils.ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        create_params = (sess, image_lists, label_name, index,
                         image_dir, category, bottleneck_dir,
                         jpeg_data_tensor, bottleneck_tensor, None)
        get_or_create_bottleneck(create_params)
        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) , ' bottleneck files created.')

def get_random_cached_bottlenecks(bottleneck_params):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    bottleneck_params - tuple of - 
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: The number of bottleneck values to return.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  (sess, image_lists, how_many, category, bottleneck_dir, image_dir,
   jpeg_data_tensor, bottleneck_tensor) = bottleneck_params
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    create_params = (sess, image_lists, label_name, image_index,
                 image_dir, category, bottleneck_dir,
                 jpeg_data_tensor, bottleneck_tensor, None)
    bottleneck = get_or_create_bottleneck(create_params)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = conts.GROUND_TRUTH_VALUE
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
    
  return (bottlenecks, ground_truths)

def get_val_test_bottlenecks(bottleneck_params):
  """Retrieves bottleneck values for cached images.
  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.
  Args:
    bottleneck_params - tuple of - 
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: The number of bottleneck values to return.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
  Returns:
    List of bottleneck arrays and their corresponding ground truths 
    and image paths.
  """
  (sess, image_lists, _, category, bottleneck_dir, image_dir,
   jpeg_data_tensor, bottleneck_tensor) = bottleneck_params
  
  class_count = len(image_lists.keys())
  # test_class_dist = get_test_random_dist(image_lists)
  bottlenecks = []
  ground_truths = []
  img_paths = []
  for label_index, img_class in enumerate(image_lists.keys()):
    if category not in image_lists[img_class]:
      tf.logging.fatal('Category does not exist %s.', category)

    if len(image_lists[img_class][category]) < 1:
      tf.logging.fatal('Category has no images - %s.', category)

    for image in image_lists[img_class][category]:
      # get bottleneck and img_path
      create_params = (sess, image_lists, img_class, None,
                 image_dir, category, bottleneck_dir,
                 jpeg_data_tensor, bottleneck_tensor, image)
      (bottleneck_values, bottleneck_path) = get_or_create_bottleneck_and_path(create_params)
      # set ground_truth
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = conts.GROUND_TRUTH_VALUE
      bottlenecks.append(bottleneck_values)
      ground_truths.append(ground_truth)
      img_paths.append(bottleneck_path)
      
  return (bottlenecks, ground_truths, img_paths)

def get_random_distorted_bottlenecks(bottleneck_params):
  """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    bottleneck_params - tuple of - 
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: The integer number of bottleneck values to return.
      category: Name string of which set of images to fetch - training, testing,
      or validation.
      image_dir: Root folder string of the subfolders containing the training
      images.
      input_jpeg_tensor: The input layer we feed the image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
  Returns:
    (bottlenecks, ground_truths) - list of bottleneck arrays and their 
                                   corresponding ground truths.
  """
  bottlenecks = []
  ground_truths = []
  (sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
   distorted_image, resized_input_tensor, bottleneck_tensor) = bottleneck_params
  class_count = len(image_lists.keys())
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    path_parameters = (image_lists, label_name, image_index,
                       image_dir, category)
    image_path = dataset.get_image_path(path_parameters)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck_params = (sess, distorted_image_data,
                         resized_input_tensor, bottleneck_tensor)
    bottleneck = run_bottleneck_on_image(bottleneck_params)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = conts.GROUND_TRUTH_VALUE
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  
  return (bottlenecks, ground_truths)
