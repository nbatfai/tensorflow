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

# Norbert Batfai, 15 Jan 2017
# nbatfai@gmail.com
# Some modifications and additions to the original code:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
# 
# This modification is described in the paper "Entropy Non-increasing Games for Improvement of Dataflow Programming"
# =============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import time

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
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
    
  batchsize = tf.Variable(100)  
  ymin = tf.reduce_min(y, 1, keep_dims=True)    
  yshift = tf.add(y, tf.abs(ymin)+.001)
  ysum = tf.reduce_sum(yshift, 1, keep_dims=True)
  prob = tf.div(yshift, ysum)
  plogp = tf.multiply(prob, tf.log(prob)/tf.log(2.0))
  plogpsum = -tf.reduce_sum(plogp, 1, keep_dims=True)
  plusminus = tf.cast(correct_prediction, tf.float32)*2.0-1.0
  plusminus = tf.reshape(plusminus, [batchsize, 1])
  info = tf.multiply(plogpsum, plusminus)
  infoacc = tf.reduce_sum(info) / tf.cast(batchsize, tf.float32)
  tf.summary.scalar("infoacc", infoacc)

  #sess = tf.InteractiveSession()  
  sess = tf.Session()
  with sess.as_default():
    
    #niter =  100000
    #batch_size =  10    
    niter = 100000
    batch_size = 100
    #niter =100000
    #batch_size =1000
    #niter =  100000
    #batch_size =  10000    
        
    merged = tf.summary.merge_all()    
    logs = "/tmp/mnist_softmax_esmu_infoacc/bs"+str(batch_size)+"_"+time.strftime("%c")
    train_writer = tf.summary.FileWriter(logs + "/train")
    test_writer = tf.summary.FileWriter(logs + "/test")
    writer = tf.summary.FileWriter(logs+"/graph")
    writer.add_graph(graph=sess.graph)
    
    tf.global_variables_initializer().run()
    print("-- Train")  
    accsum = 0.0
    pinfoaccsum = 0.0
    for i in range(niter):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      summary, acc, _, py, pysum, pyshift, pymin, pprob, pplogp, pplogpsum, pplusminus, pinfo, pinfoacc = sess.run([merged, accuracy, train_step, y, ysum, yshift, ymin, prob, plogp, plogpsum, plusminus, info, infoacc], feed_dict={batchsize: batch_size, x: batch_xs, y_: batch_ys})
      train_writer.add_summary(summary, i)
      if i % (niter/10) == 0:
	print((i/niter) * 100, "%")
      pinfoaccsum = pinfoaccsum + pinfoacc	
      accsum = accsum + acc	
    print("--     Acc: ", acc/niter)
    print("-- InfoAcc: ", pinfoaccsum/niter, " bit")
    print("----------------------------------------------------------")
    train_writer.close()

    # Test trained model
    print("-- Test")  
    summary, acc, pinfoacc = sess.run([merged, accuracy, infoacc], feed_dict={batchsize: 10000, x: mnist.test.images, y_: mnist.test.labels})  
    test_writer.add_summary(summary)
    print("--     Acc: ", acc)
    print("-- InfoAcc: ", pinfoacc, " bit")
    print("----------------------------------------------------------")
    test_writer.close()    
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
