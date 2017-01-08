#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 17:49:54 2017

Contains functions that detect centroids of annotations

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
import tensorflow as tf
import ImageAnnotation as ia
import iaConvNetTools as cn


########################################################################
### Load data
########################################################################

data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet'
ais = ia.AnnotatedImageSet()
ais.load_data_dir(data_path)


########################################################################
### Network architecture
########################################################################

x = tf.placeholder(tf.float32, [None, 3*zoom_size[0]*zoom_size[1]])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# convert input image to tensor
# x_image = [-1 x image-height x image-width x n-input-channels]
x_image_temp = tf.reshape(x, [-1,3,36,36])
x_image = tf.transpose(x_image_temp, [0,2,3,1])

# First convolutional layer
# W = [patch-height x patch-width x n-input-channels x n-output-channels])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

# convolve x_image with the weight tensor, add the bias,
# apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

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

#saver.restore(sess,"/tmp/roi_conv_model2.ckpt")

# Train for 100 steps
start = time.time()
for t in range(3000):
    if t % 10 != 0: # Just train
        samples,labels = get_roi_training_data( im0_norm, im1_norm, im2_norm,
                                                masked_roi_im, zoom_size, 200 )
        sess.run(train_step,
            feed_dict={x: samples, y_: labels, keep_prob: 0.5})
        print('.', end="", flush=True)
    else: # Dipslay progress
        samples,labels = get_roi_training_data( im0_norm, im1_norm, im2_norm,
                                                masked_roi_im, zoom_size, 200 )
        result = sess.run([merged_summary,accuracy],
            feed_dict={x: samples, y_: labels, keep_prob: 1.0})
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, t)
        end = time.time()
        print('Step {:4d}: Acc = {:6.4f} (t={})'.format(
                t, acc, str(datetime.timedelta(seconds=np.round(end-start))) ))
        save_path = saver.save(sess, "/tmp/roi_conv_model3.ckpt")
        print('  -> Model saved in file: {}'.format(save_path))
