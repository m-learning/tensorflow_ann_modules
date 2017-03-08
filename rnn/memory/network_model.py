"""
Created on Jan 27, 2017

Model implementation for QA network

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential


def _input_encoder_m(vocab_size, story_maxlen):
  """Embed the input sequence into a sequence of vectors
    Returns:
      input_encoder_m - input encoder
  """
  input_encoder_m = Sequential()
  input_encoder_m.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=story_maxlen))
  input_encoder_m.add(Dropout(0.3))
  
  return input_encoder_m
# output: (samples, story_maxlen, embedding_dim)

def _question_encoder(vocab_size, query_maxlen):
  """Embed the question into a sequence of vectors
    Returns:
      question_encoder - question encoder
  """
  question_encoder = Sequential()
  question_encoder.add(Embedding(input_dim=vocab_size,
                                 output_dim=64,
                                 input_length=query_maxlen))
  question_encoder.add(Dropout(0.3))
  
  return question_encoder
# output: (samples, query_maxlen, embedding_dim)

def _match(input_encoder_m, question_encoder):
  """Compute a 'match' between input sequence elements (which are vectors)
    and the question vector sequence
    Returns:
      match - match model
  """
  match = Sequential()
  match.add(Merge([input_encoder_m, question_encoder],
                  mode='dot',
                  dot_axes=[2, 2]))
  match.add(Activation('softmax'))
  
  return match
# output: (samples, story_maxlen, query_maxlen)

def _input_encoder_c(vocab_size, query_maxlen, story_maxlen):
  """Embed the input into a single vector with size = story_maxlen
    Returns:
      input_encoder_c - input embedded
  """
  input_encoder_c = Sequential()
  input_encoder_c.add(Embedding(input_dim=vocab_size,
                                output_dim=query_maxlen,
                                input_length=story_maxlen))
  input_encoder_c.add(Dropout(0.3))
  
  return input_encoder_c
  # output: (samples, story_maxlen, query_maxlen)
  
def _response(match, input_encoder_c):  
  """Sum the match vector with the input vector
    Returns:
      response - response model
  """
  response = Sequential()
  response.add(Merge([match, input_encoder_c], mode='sum'))
  # output: (samples, story_maxlen, query_maxlen)
  response.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)

  return response

def _answer(response, question_encoder, vocab_size):
  """Concatenate the match vector with the question vector,
    and do logistic regression on top
    Returns:
      answer - answer of model
  """
  answer = Sequential()
  answer.add(Merge([response, question_encoder], mode='concat', concat_axis=-1))
  # the original paper uses a matrix multiplication for this reduction step.
  # we choose to use a RNN instead.
  answer.add(LSTM(32))
  # one regularization layer -- more would probably be needed.
  answer.add(Dropout(0.3))
  answer.add(Dense(vocab_size))
  # we output a probability distribution over the vocabulary
  answer.add(Activation('softmax'))
  
  return answer