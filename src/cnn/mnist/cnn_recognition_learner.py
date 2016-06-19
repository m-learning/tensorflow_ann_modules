'''
Created on Jun 18, 2016

@author: Levan Tsinadze
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cnn_methods import cnn_functions
import cnn_parameters

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

class cnn_learner:
    
    def __init__(self):
        
        self.cnn_fnc = cnn_functions()
        (self.pred, self.correct_pred, self.accuracy) = self.cnn_fnc.cnn_pred()
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.cnn_fnc.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
    
    def init_mnist(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        return mnist
    
    def traint(self):
        
        # Initializing the variables
        init = tf.initialize_all_variables()
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        
        mnist = self.init_mnist()
        # Launch the graph
        with tf.Session() as sess:
            
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.cnn_fnc.x: batch_x, self.cnn_fnc.y: batch_y,
                                               self.cnn_fnc.keep_prob: cnn_parameters.CNN_DROPOUT})
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.cnn_fnc.x: batch_x,
                                                                      self.cnn_fnc.y: batch_y,
                                                                      self.cnn_fnc.keep_prob: 1.})
                    print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)
                step += 1
            print "Optimization Finished!"
            save_path = saver.save(sess, cnn_parameters.PARAMETERS_FILE_CONV_PATH)
            print "Model saved in file: %s" % save_path
        
            # Calculate accuracy for 256 mnist test images
            print "Testing Accuracy:", \
                sess.run(self.accuracy, feed_dict={self.cnn_fnc.x: mnist.test.images[:256],
                                              self.cnn_fnc.y: mnist.test.labels[:256],
                                              self.cnn_fnc.keep_prob: 1.})
                
def main():
    lrn = cnn_learner()
    lrn.traint()
    
if __name__ == "__main__":
    main()   
