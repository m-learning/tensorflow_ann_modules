"""
Created on Jun 18, 2016

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.examples.tutorials.mnist import input_data

from cnn.mnist import network_parameters as pr
from cnn.mnist.cnn_files import training_file
from cnn.mnist.network_config import cnn_functions
from cnn.utils.cnn_flags_utils import KEEP_FULL_PROB
import tensorflow as tf


# Parameters
learning_rate = 0.001
training_iters = None
batch_size = 128
display_step = 10

DECAY = 5e-4

class cnn_learner(object):
  """Training methods"""
    
  def __init__(self):
    
    self.network = cnn_functions(decay=0.00001, for_training=False)
    (self.pred, self.correct_pred, self.accuracy) = self.network.cnn_pred()
    # Define loss and optimizer
    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.network.y))
    regularizers = self.network.regularizer()
    self.cost += DECAY * regularizers
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
      
  # Initializes and gets training data
  def init_mnist(self):
    """Initializes MNIST data set
      Returns:
        MNIST - data set
    """
    self.tr_files = training_file()
    data_path = self.tr_files.get_data_directory()
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    
    return mnist
  
  def traint(self):
    """Trains neural network"""
    # Initializing the variables
    init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    
    mnist = self.init_mnist()
    parameters_path = self.tr_files.get_or_init_files_path()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(self.optimizer, feed_dict={self.network.x: batch_x, self.network.y: batch_y,
                                                self.network.keep_prob: pr.CNN_DROPOUT})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.network.x: batch_x,
                                                                  self.network.y: batch_y,
                                                                  self.network.keep_prob: KEEP_FULL_PROB})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        save_path = saver.save(sess, parameters_path)
        print("Model saved in file: %s" % save_path)
    
        # Calculate accuracy for 256 mnist test images
        print ("Testing Accuracy:", \
            sess.run(self.accuracy, feed_dict={self.network.x: mnist.test.images[:256],
                                               self.network.y: mnist.test.labels[:256],
                                               self.network.keep_prob: KEEP_FULL_PROB}))
                
def main():
  """Runs training with graph initialization"""
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--training_steps',
                          type=int,
                          default=200000,
                          help='Number of training iterations')
  (argument_flags, _) = arg_parser.parse_known_args()
  global training_iters
  training_iters = argument_flags.training_steps
  
  with tf.Graph().as_default():
      learner = cnn_learner()
      learner.traint()
    
if __name__ == '__main__':
  main()   
