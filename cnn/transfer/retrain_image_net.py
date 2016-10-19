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
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

import cnn.transfer.training_flags_mod as training_flags_mod
import  cnn.transfer.graph_config as graph_config
import cnn.transfer.config_image_net as config
import  cnn.transfer.bottleneck_config as bottleneck
import  cnn.transfer.distort_config as distort

# Add final training operations
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  bottleneck_input = tf.placeholder_with_default(
      bottleneck_tensor, shape=[None, graph_config.BOTTLENECK_TENSOR_SIZE],
      name='BottleneckInputPlaceholder')
  layer_weights = tf.Variable(
      tf.truncated_normal([graph_config.BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
      name='final_weights')
  layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
  logits = tf.matmul(bottleneck_input, layer_weights,
                     name='final_matmul') + layer_biases
  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  ground_truth_input = tf.placeholder(tf.float32,
                                      [None, class_count],
                                      name='GroundTruthInput')
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_input)
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  train_step = tf.train.GradientDescentOptimizer(training_flags_mod.learning_rate).minimize(cross_entropy_mean)
  
  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Nothing.
  """
  correct_prediction = tf.equal(tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  
  return evaluation_step

# Saves trained weights and biases
def save_trained_parameters(sess, graph, keys):
  """
    Saves trained checkpoint
    Args:
     sess - TensorFlow session
     graph - Inception-V3 graph
     keys - keys to save checkpoint
  """
  
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [training_flags_mod.final_tensor_name])
  with gfile.FastGFile(tr_flags.output_graph, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  with gfile.FastGFile(tr_flags.output_labels, 'w') as f:
    f.write('\n'.join(keys) + '\n')

# Test neural net
def test_trained_network(sess, validation_parameters):
  
  (_, image_lists, _, _, _, bottleneck_tensor,
   jpeg_data_tensor, _, bottleneck_input,
   ground_truth_input, evaluation_step, _) = validation_parameters
  
  test_bottlenecks, test_ground_truth = bottleneck.get_random_cached_bottlenecks(
      sess, image_lists, training_flags_mod.test_batch_size, 'testing',
      tr_flags.bottleneck_dir, tr_flags.image_dir, jpeg_data_tensor,
      bottleneck_tensor)
  test_accuracy = sess.run(
      evaluation_step,
      feed_dict={bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth})
  print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
  

# Iterates and trains neural network
def iterate_and_train(sess, iteration_parameters):
  
  (do_distort_images, image_lists,
   distorted_jpeg_data_tensor, distorted_image_tensor,
   resized_image_tensor, bottleneck_tensor, jpeg_data_tensor,
   train_step, bottleneck_input, ground_truth_input,
   evaluation_step, cross_entropy) = iteration_parameters
   
  
  # Run the training for as many cycles as requested on the command line.
  for i in range(training_flags_mod.how_many_training_steps):
    # Get a catch of input bottleneck values, either calculated fresh every time
    # with distortions applied, or from the cache stored on disk.
    if do_distort_images:
      train_bottlenecks, train_ground_truth = bottleneck.get_random_distorted_bottlenecks(
          sess, image_lists, training_flags_mod.train_batch_size, 'training',
          tr_flags.image_dir, distorted_jpeg_data_tensor,
          distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
    else:
      train_bottlenecks, train_ground_truth = bottleneck.get_random_cached_bottlenecks(
          sess, image_lists, training_flags_mod.train_batch_size, 'training',
          tr_flags.bottleneck_dir, tr_flags.image_dir, jpeg_data_tensor,
          bottleneck_tensor)
    # Feed the bottlenecks and ground truth into the graph, and run a training
    # step.
    sess.run(train_step,
             feed_dict={bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth})
    # Every so often, print out how well the graph is training.
    is_last_step = (i + 1 == training_flags_mod.how_many_training_steps)
    if (i % training_flags_mod.eval_step_interval) == 0 or is_last_step:
      train_accuracy, cross_entropy_value = sess.run(
          [evaluation_step, cross_entropy],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                      train_accuracy * 100))
      print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                 cross_entropy_value))
      validation_bottlenecks, validation_ground_truth = (
          bottleneck.get_random_cached_bottlenecks(
              sess, image_lists, training_flags_mod.validation_batch_size, 'validation',
              tr_flags.bottleneck_dir, tr_flags.image_dir, jpeg_data_tensor,
              bottleneck_tensor))
      validation_accuracy = sess.run(
          evaluation_step,
          feed_dict={bottleneck_input: validation_bottlenecks,
                     ground_truth_input: validation_ground_truth})
      print('%s: Step %d: Validation accuracy = %.1f%%' % 
            (datetime.now(), i, validation_accuracy * 100))

# Validates and prepares training parameters
def prepare_parameters(tr_file):
  
  # Set up flags and training data
  global tr_flags
  tr_flags = config.init_flags_and_files(tr_file)
  
  # Set up the pre-trained graph.
  config.maybe_download_and_extract()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      graph_config.create_inception_graph(tr_flags))

  # Look at the folder structure, and create lists of all the images.
  image_lists = config.create_image_lists(tr_flags.image_dir, training_flags_mod.testing_percentage,
                                   training_flags_mod.validation_percentage)
  print(image_lists)
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + tr_flags.image_dir)
    return None
  if class_count == 1:
    print('Only one valid folder of images found at ' + tr_flags.image_dir + 
          ' - multiple classes are needed for classification.')
    return None
  
  return (graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor, image_lists)

# Set up all our weights to their initial default values.
def prepare_session(sess):
    
  init = tf.initialize_all_variables()
  sess.run(init)

# Prepares training iteration parameters
def prepare_iteration_parameters(prepared_parameters):
  
  (graph, bottleneck_tensor, jpeg_data_tensor,
   resized_image_tensor, image_lists) = prepared_parameters

  # See if the command-line flags mean we're applying any distortions.
  (sess, do_distort_images,
   distorted_jpeg_data_tensor, distorted_image_tensor) = distort.distort_images(prepared_parameters, tr_flags)
  # Add the new layer that we'll be training.
  (train_step, cross_entropy, bottleneck_input,
   ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                          training_flags_mod.final_tensor_name,
                                          bottleneck_tensor)
  # Set up all our weights to their initial default values.
  prepare_session(sess)

  # Create the operations we need to evaluate the accuracy of our new layer.
  evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

  return (sess, graph, (do_distort_images, image_lists,
                          distorted_jpeg_data_tensor, distorted_image_tensor,
                          resized_image_tensor, bottleneck_tensor, jpeg_data_tensor,
                          train_step, bottleneck_input, ground_truth_input,
                          evaluation_step, cross_entropy))

# Validates tests and saves neural network graph
def validate_test_and_save(sess, graph, validation_parameters):
  
  (_, image_lists, _, _, _, _, _, _, _, _, _, _) = validation_parameters
  test_trained_network(sess, validation_parameters)

  # Write out the trained graph and labels with the weights stored as constants.
  save_trained_parameters(sess, graph, image_lists.keys())

# Retrains neural network after validation
def retrain_valid_net(prepared_parameters):
  
  # Prepares training evaluation and test parameters
  (sess, graph, iteration_parameters) = prepare_iteration_parameters(prepared_parameters)
  # Run the training for as many cycles as requested on the command line.
  iterate_and_train(sess, iteration_parameters)

  # We've completed all our training, so run a final test evaluation on
  # some new images we haven't used before.
  validate_test_and_save(sess, graph, iteration_parameters)

# Runs training and testing
def retrain_net(tr_file):
  """
    Retrains Inception on different data set
    Args:
      tr_file - utility object to manage files
  """
  
  # Prepares training parameters
  prepared_parameters = prepare_parameters(tr_file)
  
  if prepared_parameters is None:
    return -1
  else:
    retrain_valid_net(prepared_parameters)
