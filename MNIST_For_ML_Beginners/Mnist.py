from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""


import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# =============================================================================
# 
# 
# 
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# import matplotlib.pyplot as plt
# import numpy as np
# import random as ran
# import tensorflow as tf
# 
# 
# def TRAIN_SIZE(num):
#     print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
#     print ('--------------------------------------------------')
#     x_train = mnist.train.images[:num,:]
#     print ('x_train Examples Loaded = ' + str(x_train.shape))
#     y_train = mnist.train.labels[:num,:]
#     print ('y_train Examples Loaded = ' + str(y_train.shape))
#     print('')
#     return x_train, y_train
# 
# def TEST_SIZE(num):
#     print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
#     print ('--------------------------------------------------')
#     x_test = mnist.test.images[:num,:]
#     print ('x_test Examples Loaded = ' + str(x_test.shape))
#     y_test = mnist.test.labels[:num,:]
#     print ('y_test Examples Loaded = ' + str(y_test.shape))
#     return x_test, y_test
# 
# # Display Images
# def display_digit(num):
#     print(y_train[num])
#     label = y_train[num].argmax(axis=0)
#     image = x_train[num].reshape([28,28])
#     plt.title('Example: %d  Label: %d' % (num, label))
#     plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#     plt.show()
# 
# def display_mult_flat(start, stop):
#     images = x_train[start].reshape([1,784])
#     for i in range(start+1,stop):
#         images = np.concatenate((images, x_train[i].reshape([1,784])))
#     plt.imshow(images, cmap=plt.get_cmap('gray_r'))
#     plt.show()
#   
# x_train, y_train = TRAIN_SIZE(55000)
# 
# 
# display_digit(ran.randint(0, x_train.shape[0]))
# display_mult_flat(0,400)
# 
# # TenserFlow Part now
# 
# 
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# 
# 
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# 
# 
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# 
# y = tf.nn.softmax(tf.matmul(x,W) + b)
# 
# print(y)
# 
# x_train, y_train = TRAIN_SIZE(3)
# sess.run(tf.global_variables_initializer())
# print(sess.run(y, feed_dict={x: x_train}))
# 
# sess.run(tf.nn.softmax(tf.zeros([4])))
# sess.run(tf.nn.softmax(tf.constant([0.1, 0.005, 2])))
# 
# 
# #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 
# 
# x_train, y_train = TRAIN_SIZE(5500)
# x_test, y_test = TEST_SIZE(10000)
# LEARNING_RATE = 0.5
# TRAIN_STEPS = 2500
# 
# 
# init = tf.global_variables_initializer()
# sess.run(init)
# 
# 
# 
# training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 
# 
# 
# for i in range(TRAIN_STEPS+1):
#     sess.run(training, feed_dict={x: x_train, y_: y_train})
#     if i%100 == 0:
#         print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))
#         
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     weight = sess.run(W)[:,i]
#     plt.title(i)
#     plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
#     frame1 = plt.gca()
#     frame1.axes.get_xaxis().set_visible(False)
#     frame1.axes.get_yaxis().set_visible(False)
#     
# plt.show()
# 
# 
# # =============================================================================
# # x_train, y_train = TRAIN_SIZE(1)
# # display_digit(0)
# # =============================================================================
# 
# answer = sess.run(y, feed_dict={x: x_train})
# print(answer)
# 
# answer.argmax()
# 
# 
# def display_compare(num):
#     # THIS WILL LOAD ONE TRAINING EXAMPLE
#     x_train = mnist.train.images[num,:].reshape(1,784)
#     y_train = mnist.train.labels[num,:]
#     # THIS GETS OUR LABEL AS A INTEGER
#     label = y_train.argmax()
#     # THIS GETS OUR PREDICTION AS A INTEGER
#     prediction = sess.run(y, feed_dict={x: x_train}).argmax()
#     plt.title('Prediction: %d Label: %d' % (prediction, label))
#     plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
#     plt.show()
#     
#     
# display_compare(ran.randint(0, 55000))
# 
# =============================================================================
