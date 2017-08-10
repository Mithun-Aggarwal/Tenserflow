# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:17:30 2017

@author: ksxl806
"""

# Import `tensorflow`
import tensorflow as tf


config=tf.ConfigProto(log_device_placement=True)

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(result)
  print(output)