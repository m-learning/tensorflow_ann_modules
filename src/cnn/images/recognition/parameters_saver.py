'''
Created on Jun 15, 2016

@author: levan Tsinadze
'''

import os

parameters_folder = '/storage/ann/digits/convolution'
parameters_file = parameters_folder + '/' + 'model.ckpt'
parameters_file_conv = '/storage/ann/digits/parameters/conv_model.ckpt'
parameters_file_conv_saved = '/storage/ann/digits/parameters_saved/conv_model.ckpt'
if not os.path.exists(parameters_file):
    fl = file(parameters_file, 'w')
ls = os.listdir(parameters_folder)
print ls

def save_model(saver, sess):
    save_path = saver.save(sess, parameters_file)
    print("Model saved in file: %s" % save_path)
