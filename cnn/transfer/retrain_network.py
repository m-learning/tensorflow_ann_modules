# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple transfer learning with an Inception v3 architecture model.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:

bazel build third_party/tensorflow/examples/image_retraining:retrain && \
bazel-bin/third_party/tensorflow/examples/image_retraining/retrain \
--image_dir ~/flower_photos

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from  cnn.transfer import graph_config as gconf
from cnn.transfer import dataset_config as dataset
import cnn.transfer.board_logger as logger
import  cnn.transfer.bottleneck_config as bottleneck
import  cnn.transfer.distort_config as distort
import cnn.transfer.network_config as config
import cnn.transfer.training_flags as flags
import tensorflow as tf


VALID_RESULT_CODE = 0
ERROR_RESULT_CODE = -1

TRAINING_CATEGORY = 'training'
VALIDATION_CATEGORY = 'validation'
TESTING_CATEGORY = 'testing'

def save_trained_parameters(sess, graph, keys):
  """Saves trained checkpoint
    Args:
     sess - current TensorFlow session
     graph - Inception-V3 graph
     keys - keys to save checkpoint
  """
  
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [flags.final_tensor_name])
  with gfile.FastGFile(flags.output_graph, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  with gfile.FastGFile(flags.output_labels, 'w') as f:
    f.write('\n'.join(keys) + '\n')

def test_trained_network(sess, validation_parameters):
  """Tests trained network on test set
    Args:
      sess - current TensorFlow session
      validation_parameters - parameters for test 
      and validation
  """
  
  (_, image_lists, _, _, _, bottleneck_tensor,
   jpeg_data_tensor, _, bottleneck_input,
   ground_truth_input, keep_prob, evaluation_step, prediction_step, _, _) = validation_parameters
  bottleneck_params = (sess, image_lists, flags.test_batch_size, TESTING_CATEGORY,
                       flags.bottleneck_dir, flags.image_dir, jpeg_data_tensor,
                       bottleneck_tensor)
  (test_bottlenecks, test_ground_truth, _) = bottleneck.get_val_test_bottlenecks(bottleneck_params)
  test_accuracy, _ = sess.run(
      [evaluation_step, prediction_step],
      feed_dict={bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth,
                 keep_prob: flags.keep_all_prob})
  print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
  

def iterate_and_train(sess, iteration_parameters):
  """Trains network with additional parameters
    Args:
      sess - current TensorFlow session
      iteration_parameters - additional training parameters
  """
  
  (do_distort_images, image_lists,
   distorted_jpeg_data_tensor, distorted_image_tensor,
   resized_image_tensor, bottleneck_tensor, jpeg_data_tensor,
   train_step, bottleneck_input, ground_truth_input, keep_prob,
   evaluation_step, _, cross_entropy, _) = iteration_parameters
   
  # Merge all the summaries and write them out to /tmp/retrain_inception_logs (by default)
  (merged, train_writer, validation_writer) = logger.init_writer(sess)
  
  # Run the training for as many cycles as requested on the command line.
  validation_bottlenecks = None
  validation_ground_truth = None
  for i in range(flags.how_many_training_steps):
    # Get a catch of input bottleneck values, either calculated fresh every time
    # with distortions applied, or from the cache stored on disk.
    if do_distort_images:
      bottleneck_params = (sess, image_lists, flags.train_batch_size, TRAINING_CATEGORY,
                           flags.image_dir, distorted_jpeg_data_tensor,
                           distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      train_bottlenecks, train_ground_truth = bottleneck.get_random_distorted_bottlenecks(bottleneck_params)
    else:
      bottleneck_params = (sess, image_lists, flags.train_batch_size, TRAINING_CATEGORY,
                           flags.bottleneck_dir, flags.image_dir, jpeg_data_tensor, bottleneck_tensor)
      (train_bottlenecks, train_ground_truth) = bottleneck.get_random_cached_bottlenecks(bottleneck_params)
    # Every so often, print out how well the graph is training.
    is_last_step = (i + 1 == flags.how_many_training_steps)
    if (i % flags.eval_step_interval) == 0 or is_last_step:
      train_accuracy, cross_entropy_value = sess.run(
          [evaluation_step, cross_entropy],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth,
                     keep_prob: flags.keep_prob})
      print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
      print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
      bottleneck_params = (sess, image_lists, flags.validation_batch_size, VALIDATION_CATEGORY,
                           flags.bottleneck_dir, flags.image_dir, jpeg_data_tensor, bottleneck_tensor)
      if validation_bottlenecks is None or validation_ground_truth is None:
        (validation_bottlenecks, validation_ground_truth, _) = bottleneck.get_val_test_bottlenecks(bottleneck_params)
      # Run a validation step and capture training summaries for TensorBoard with the `merged` op.
      validation_summary, validation_accuracy = sess.run(
          [merged, evaluation_step],
          feed_dict={bottleneck_input: validation_bottlenecks,
                     ground_truth_input: validation_ground_truth,
                     keep_prob: flags.keep_all_prob})
      validation_writer.add_summary(validation_summary, i)
      print('%s: Step %d: Validation accuracy = %.1f%%' % 
            (datetime.now(), i, validation_accuracy * 100))
    else:
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run([merged, train_step],
               feed_dict={bottleneck_input: train_bottlenecks,
                          ground_truth_input: train_ground_truth,
                          keep_prob: flags.keep_prob})
      train_writer.add_summary(train_summary, i)
      

def prepare_parameters(tr_file):
  """Prepares training parameters
    Args:
      tr_file - file management utility
    Returns:
      tuple of -
        graph - network graph
        bottleneck_tensor - bottleneck as tensor
        jpeg_data_tensor - image as tensor
        resized_image_tensor - resized image as tensor
        image_lists - training images
  """
  
  # Configures training flags 
  config.init_flags_and_files(tr_file)
  # Set up the pre-trained graph.
  dataset.maybe_download_and_extract()
  (graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor) = gconf.create_inception_graph()

  # Look at the folder structure, and create lists of all the images.
  image_lists = dataset.create_image_lists(flags.image_dir, flags.testing_percentage,
                                          flags.validation_percentage)
  print(image_lists)
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + flags.image_dir)
    return None
  if class_count == 1:
    print('Only one valid folder of images found at ' + flags.image_dir + 
          ' - multiple classes are needed for classification.')
    return None
  
  return (graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor, image_lists)

def prepare_session(sess):
  """Prepares session for training, validation 
    and test steps
    Args:
      sess - current TensorFlow session
    """
  init = tf.initialize_all_variables()
  sess.run(init)

def prepare_iteration_parameters(prepared_parameters):
  """Prepares parameters for training iterations
    Args:
      prepared_parameters - prepared training 
      parameters
    Returns:
      sess - current TensorFlow session
      graph - network graph
      do_distort_images - distort flag
      image_lists - training images list
      distorted_jpeg_data_tensor - distorted JPEG image 
                                   as tensor
      distorted_image_tensor - distorted image 
                               as tensor
      bottleneck_tensor - bottleneck layer tensor
      jpeg_data_tensor - images for iteration
      train_step - training step descriptor
      bottleneck_input - input for bottleneck layer
      ground_truth_input - label for input
      evaluation_step - prepared step for evaluation
      cross_entropy - result producer function                             
  """
  
  (graph, bottleneck_tensor, jpeg_data_tensor,
   resized_image_tensor, image_lists) = prepared_parameters

  # See if the command-line flags mean we're applying any distortions.
  (sess, do_distort_images,
   distorted_jpeg_data_tensor, distorted_image_tensor) = distort.distort_images(prepared_parameters)
  # Add the new layer that we'll be training.
  num_classes = len(image_lists.keys())  # Calculates number of output classes
  (train_step, cross_entropy, total_loss, bottleneck_input,
   ground_truth_input, final_tensor, keep_prob) = config.add_final_training_ops(num_classes,
                                                              flags.final_tensor_name,
                                                              bottleneck_tensor)
  # Set up all our weights to their initial default values.
  prepare_session(sess)
  # Create the operations we need to evaluate the accuracy of our new layer.
  evaluation_step = config.add_evaluation_step(final_tensor, ground_truth_input)
  prediction_step = config.add_prediction_step(final_tensor)

  return (sess, graph, (do_distort_images, image_lists,
                          distorted_jpeg_data_tensor, distorted_image_tensor,
                          resized_image_tensor, bottleneck_tensor, jpeg_data_tensor,
                          train_step, bottleneck_input, ground_truth_input, keep_prob,
                          evaluation_step, prediction_step, cross_entropy, total_loss))

def validate_test_and_save(sess, graph, validation_parameters):
  """Validates and / or tests network
    Args:
      sess - current TensorFlow session
      graph - network graph
      validation_parameters - prepared parameters 
                              for validation
  """
  
  (_, image_lists, _, _, _, _, _, _, _, _, _, _, _, _, _) = validation_parameters
  test_trained_network(sess, validation_parameters)
  # Write out the trained graph and labels with the weights stored as constants.
  save_trained_parameters(sess, graph, image_lists.keys())

def retrain_valid_net(prepared_parameters):
  """Retrains Inception after validation over parameters
    Args:
      prepared_parameters - tuple of training parameters
  """
  
  # Prepares training evaluation and test parameters
  (sess, graph, iteration_parameters) = prepare_iteration_parameters(prepared_parameters)
  # Run the training for as many cycles as requested on the command line.
  iterate_and_train(sess, iteration_parameters)

  # We've completed all our training, so run a final test evaluation on
  # some new images we haven't used before.
  validate_test_and_save(sess, graph, iteration_parameters)

def retrain_net(tr_file):
  """Retrains Inception on different data set
    Args:
      tr_file - utility object to manage files
  """
  # Prepares training parameters
  prepared_parameters = prepare_parameters(tr_file)
  if prepared_parameters is not None:
    retrain_valid_net(prepared_parameters)
    result_code = VALID_RESULT_CODE
  else:
    result_code = ERROR_RESULT_CODE
  
  return result_code
