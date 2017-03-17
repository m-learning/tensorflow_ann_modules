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


def train_model(flags):
  """Trains network and saves weights"""
  
  ((X_train, y_train), _) = preprocessor.init_training_set(flags)
  model = network.prepare_for_train(flags, is_training=True)
  model.fit(X_train, y_train, epochs=3, batch_size=64)
  weights_path = config.init_weights_path()
  model.save_weights(weights_path)
  
  
def evaluate_model(flags):
  """Retrieves trained weights and evaluates network model"""
  
  weights_path = config.init_weights_path()
  (_, (X_test, y_test)) = preprocessor.init_training_set(flags)
  model = network.init_model(flags, is_training=False)
  model.load_weights(weights_path)
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
  """Trains network model"""
  
  flags = config.parse_args()
  if flags.train:
    train_model(flags)
  else:
    evaluate_model(flags)