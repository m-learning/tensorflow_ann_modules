'''
Created on Jun 25, 2016
Recognizes for image
@author: Levan Tsinadze
'''
import tensorflow as tf
from cnn.mnist.cnn_input_reader import read_input_file
from cnn_files import training_file
from cnn_methods import cnn_functions

class image_recognizer:
    
    
        def recognize_image(self, image_file_path):
        
            cnn_fnc = cnn_functions()
            pred = cnn_fnc.conv2d
            
            # Evaluate model
            recognize_image = tf.argmax(pred, 1)
            # Initializing saver to read trained data
            saver = tf.train.Saver()
            tf.initialize_all_variables()
            tr_files = training_file()
            model_path = tr_files.get_or_init_files_path()
            with tf.Session() as sess:
                print 'Start session'
                # Initialize variables
                saver.restore(sess, model_path)
                print "Model restored from file: %s" % model_path
                image_rec = read_input_file(image_file_path)
                # Recognize image
                resp_dgt = sess.run(recognize_image, feed_dict={self.x: image_rec,
                                                  self.keep_prob: 0.75})
                print "Recognized image:", resp_dgt[0]
            
            return resp_dgt[0]
    
    