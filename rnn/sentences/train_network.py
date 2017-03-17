"""
Created on Mar 17, 2017

Trains and evaluates network model

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rnn.sentences import data_prepocessor as preprocessor
from rnn.sentences import network_model as network
from rnn.sentences import training_flags as config
from rnn.sentences.training_flags import flags


def train_model():
  """Trains network and saves weights"""
  
  ((X_train, y_train), _) = preprocessor.init_training_set()
  model = network.prepare_for_train(flags, is_training=True)
  model.fit(X_train, y_train, epochs=3, batch_size=64)
  weights_path = config.init_weights_path()
  model.save_weights(weights_path)
  
  
def evaluate_model():
  """Retrieves trained weights and evaluates network model"""
  
  weights_path = config.init_weights_path()
  (_, (X_test, y_test)) = preprocessor.init_training_set()
  model = network.init_model(flags, is_training=False)
  model.load_weights(weights_path)
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1] * 100))
