#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 8 12:15:25 2017

Contains functions that set up a convolutional neural net for image annotation

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
import tensorflow as tf


########################################################################
### Supporting functions
########################################################################

class Conv2Net(object):
    """Holds a 2 layer convolutional neural network for annotating
    multi channel images"""

    def __init__(self, image_size, n_channels, output_size):
        """Initializes all variables and sets up the network"""
        self.y_res = image_size[0]
        self.x_res = image_size[1]
        self.n_input_channels = n_channels
        self.out_y_res = output_size[0]
        self.out_x_res = output_size[1]
        self.conv1_size = 5
        self.conv1_n_chan = 32
        self.conv1_n_pool = 2
        self.conv2_size = 5
        self.conv2_n_chan = 64
        self.conv2_n_pool = 2

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, shape = [None,
            self.out_y_res * self.out_x_res] )

        # Convert input image to tensor with channel as last dimension
        # x_image = [-1 x im-height x im-width x n-input-channels]
        x_image_temp = tf.reshape(self.x, [-1,
            self.n_channels,self.y_res,self.x_res])
        x_image = tf.transpose(x_image_temp, [0,2,3,1])

        #########################################################
        # Set up convolutional layer 1
        # W = [im-height x im-width x n-input-channels x n-output-channels])
        conv1_shape = [self.conv1_size, self.conv1_size,
                       self.n_input_channels, self.conv1_n_chan]
        W_conv1 = tf.Variable( tf.truncated_normal(
                               shape=conv1_shape, stddev=0.1))
        b_conv1 = tf.Variable( tf.constant( 0.1, shape=self.conv1_n_chan ) )

        # Convolve x_image with the weight tensor
        conv1_lin = tf.nn.conv2d( x_image, W_conv1,
                                  strides=[1, 1, 1, 1], padding='SAME' )

        # Add bias and apply transfer function
        conv1_relu = tf.nn.relu( conv1_lin + b_conv1 )

        # Max pooling
        kernel = [1, self.conv1_n_pool, self.conv1_n_pool, 1]
        conv1_pool = tf.nn.max_pool( conv1_relu,
                        ksize=kernel, strides=kernel, padding='SAME' )

        #########################################################
        # Convolutional layer 2
        conv2_shape = [self.conv2_size, self.conv2_size,
                       self.conv1_n_chan, self.conv2_n_chan]
        W_conv2 = tf.Variable( tf.truncated_normal(
                               shape=conv2_shape, stddev=0.1 ) )
        b_conv2 = tf.Variable( tf.constant( 0.1, shape=self.conv2_n_chan ) )

        # Convolve x_image with the weight tensor
        conv2_lin = tf.nn.conv2d( conv1_pool, W_conv2,
                                  strides=[1, 1, 1, 1], padding='SAME' )

        # Add bias and apply transfer function
        conv2_relu = tf.nn.relu( conv2_lin + b_conv2 )

        # Max pooling
        kernel = [1, self.conv2_n_pool, self.conv2_n_pool, 1]
        conv2_pool = tf.nn.max_pool( conv2_relu,
                        ksize=kernel, strides=kernel, padding='SAME' )

        #########################################################
        # Densely Connected Layer
        W_fc1 = weight_variable([9 * 9 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 9*9*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout during training will prevent overfitting
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout layer
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)

        # Define how to test trained model
        network_prediction  = tf.cast( tf.argmax(y_conv,1), tf.float32 )
        correct_prediction = tf.equal( tf.argmax(y_conv,1), tf.argmax(y_,1) )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create saver op
        saver = tf.train.Saver()

        # Initialize and start session
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        # Set up tensorboard
        accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        merged_summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/roiconv_logs/run6", sess.graph)


    # Weight initialization function
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # Bias initialization function
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # Convolution
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # Max pooling
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
