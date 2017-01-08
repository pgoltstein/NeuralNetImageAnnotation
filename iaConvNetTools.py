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
import time, datetime


########################################################################
### Supporting functions
########################################################################

class ConvNetCnv2Fc1(object):
    """Holds a convolutional neural network for annotating
    multi channel images.
    2 convolutional layers, 1 fully connected layer, 1 output layer"""

    def __init__(self, input_image_size, n_input_channels, output_size,
                conv1_size=5, conv1_n_chan=32, conv1_n_pool=2,
                conv2_size=5, conv2_n_chan=64, conv2_n_pool=2,
                fc1_n_chan=1024, fc1_dropout=0.5 ):
        """Initializes all variables and sets up the network
        input_image_size: Tuple containing (y,x) size of input image
        output_image_size: Tuple containing dimensions of network output"""
        self.y_res = input_image_size[0]
        self.x_res = input_image_size[1]
        self.n_input_channels = n_input_channels
        self.out_y_res = output_size[0]
        self.out_x_res = output_size[1]
        self.conv1_size = conv1_size
        self.conv1_n_chan = conv1_n_chan
        self.conv1_n_pool = conv1_n_pool
        self.conv2_size = conv2_size
        self.conv2_n_chan = conv2_n_chan
        self.conv2_n_pool = conv2_n_pool
        self.fc1_y_size = int( np.ceil( np.ceil(
            self.y_res/self.conv1_n_pool ) / self.conv2_n_pool ) )
        self.fc1_x_size = int( np.ceil( np.ceil(
            self.x_res/self.conv1_n_pool ) / self.conv2_n_pool ) )
        self.fc1_n_chan = fc1_n_chan
        self.fc1_dropout = fc1_dropout

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, shape = [None,
            self.out_y_res * self.out_x_res] )

        # Convert input image to tensor with channel as last dimension
        # x_image = [-1 x im-height x im-width x n-input-channels]
        x_image_temp = tf.reshape(self.x, [-1,
            self.n_input_channels,self.y_res,self.x_res])
        x_image = tf.transpose(x_image_temp, [0,2,3,1])

        #########################################################
        # Set up convolutional layer 1
        # W = [im-height x im-width x n-input-channels x n-output-channels])
        self.conv1_shape = [self.conv1_size, self.conv1_size,
                       self.n_input_channels, self.conv1_n_chan]
        self.W_conv1 = tf.Variable( tf.truncated_normal(
                               shape=self.conv1_shape, stddev=0.1))
        self.b_conv1 = tf.Variable( tf.constant(0.1,
                                                shape=[self.conv1_n_chan] ))

        # Convolve x_image with the weight tensor
        self.conv1_lin = tf.nn.conv2d( x_image, self.W_conv1,
                                  strides=[1, 1, 1, 1], padding='SAME' )

        # Add bias and apply transfer function
        self.conv1_relu = tf.nn.relu( self.conv1_lin + self.b_conv1 )

        # Max pooling
        self.conv1_kernel = [1, self.conv1_n_pool, self.conv1_n_pool, 1]
        self.conv1_pool = tf.nn.max_pool( self.conv1_relu,
            ksize=self.conv1_kernel, strides=self.conv1_kernel, padding='SAME')

        #########################################################
        # Convolutional layer 2
        self.conv2_shape = [self.conv2_size, self.conv2_size,
                       self.conv1_n_chan, self.conv2_n_chan]
        self.W_conv2 = tf.Variable( tf.truncated_normal(
                               shape=self.conv2_shape, stddev=0.1 ) )
        self.b_conv2 = tf.Variable( tf.constant(0.1,
                                                shape=[self.conv2_n_chan] ))

        # Convolve x_image with the weight tensor
        self.conv2_lin = tf.nn.conv2d( self.conv1_pool, self.W_conv2,
                                  strides=[1, 1, 1, 1], padding='SAME' )

        # Add bias and apply transfer function
        self.conv2_relu = tf.nn.relu( self.conv2_lin + self.b_conv2 )

        # Max pooling
        self.conv2_kernel = [1, self.conv2_n_pool, self.conv2_n_pool, 1]
        self.conv2_pool = tf.nn.max_pool( self.conv2_relu,
            ksize=self.conv2_kernel, strides=self.conv2_kernel, padding='SAME')


        #########################################################
        # Densely Connected Layer
        # Weights and bias
        self.fc1_shape = [self.fc1_y_size * self.fc1_x_size * self.conv2_n_chan,
                          self.fc1_n_chan]
        self.W_fc1 = tf.Variable( tf.truncated_normal(
                               shape=self.fc1_shape, stddev=0.1 ) )
        self.b_fc1 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Flatten output from conv2
        self.conv2_pool_flat = tf.reshape(
            self.conv2_pool, [-1, self.fc1_shape[0]] )

        # Calculate network step
        self.fc1_relu = tf.nn.relu( tf.matmul( self.conv2_pool_flat,
            self.W_fc1) + self.b_fc1 )

        # Set up dropout option for fc1
        self.fc1_keep_prob = tf.placeholder(tf.float32)
        self.fc1_relu_drop = tf.nn.dropout(self.fc1_relu, self.fc1_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = [self.fc1_n_chan, self.out_y_res*self.out_x_res]
        self.W_fc_out = tf.Variable( tf.truncated_normal(
                                shape=self.fc_out_shape, stddev=0.1 ) )
        self.b_fc_out = tf.Variable( tf.constant(0.1,
                                shape=[self.fc_out_shape[1]] ))

        # Calculate network step
        self.fc_out_lin = tf.matmul( self.fc1_relu_drop,
                                     self.W_fc_out ) + self.b_fc_out

        #########################################################
        # Define cost function and optimizer algorithm
        self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                                            self.fc_out_lin, self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(5e-4).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.network_prediction  = tf.cast( tf.argmax(
                                        self.fc_out_lin, 1 ), tf.float32 )
        self.is_correct_prediction = tf.equal( tf.argmax( self.fc_out_lin, 1 ),
                                               tf.argmax( self.y_trgt, 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast(
                                    self.is_correct_prediction, tf.float32 ) )

    def start(self):
        """Initializes all variables and starts session"""
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def test(self, test_image_set, step_no, t_start, m_samples=100):
        # Get m samples and labels from a single AnnotatedImage
        samples,labels = \
            test_image_set.centroid_detection_sample(
                m_samples=m_samples, zoom_size=(self.y_res,self.y_res) )

        # Calculate network accuracy
        result = self.sess.run( [self.accuracy], feed_dict={
            self.x: samples, self.y_trgt: labels,
            self.fc1_keep_prob: 1.0 })
        acc = result[0]
        t_curr = time.time()
        print('\nStep {:4d}: Acc = {:6.4f} (t={})'.format( step_no, acc,
            str(datetime.timedelta(seconds=np.round(t_curr-t_start))) ),
            end="", flush=True)

    def train(self, training_image_set, m_samples=100, n_epochs=100,
                    display_every_n=10):
        """Trains network on training_image_set"""
        t_start = time.time()
        print("\nStart training network @ {}".format(
            str(datetime.timedelta(seconds=np.round(t_start))) ) )

        # Loop across training epochs
        for step_no in range(n_epochs):
            # Dipslay progress if necessary
            if step_no % display_every_n == 0:
                self.test(training_image_set, step_no, t_start, m_samples)

            # Train the network on m samples and labels
            samples,labels = \
                training_image_set.centroid_detection_sample(
                    m_samples=m_samples, zoom_size=(self.y_res,self.y_res) )
            self.sess.run( self.train_step, feed_dict={ self.x: samples,
                self.y_trgt: labels, self.fc1_keep_prob: self.fc1_dropout } )
            print('.', end="", flush=True)
        # Dipslay final performance
        self.test(training_image_set, step_no, t_start, m_samples=1000)
        print("\n")
