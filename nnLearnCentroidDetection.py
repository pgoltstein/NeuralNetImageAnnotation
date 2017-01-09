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
# Load data

training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
cv_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet2'
zoom_size = (36,36)
training_image_set = ia.AnnotatedImageSet()
training_image_set.load_data_dir(training_data_path)
cv_image_set = ia.AnnotatedImageSet()
cv_image_set.load_data_dir(cv_data_path)
# small_training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet_small'
# small_training_image_set = ia.AnnotatedImageSet()
# small_training_image_set.load_data_dir(small_training_data_path)

########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1(
        input_image_size=zoom_size, n_input_channels=3, output_size=(1,2),
        conv1_size=7, conv1_n_chan=32, conv1_n_pool=2,
        conv2_size=7, conv2_n_chan=64, conv2_n_pool=2,
        fc1_n_chan=512, fc1_dropout=0.5, alpha=5e-4 )
nn.start()
# nn.load_network_parameters('centroid_net_5x5_2',training_data_path)
# nn.load_network_parameters('centroid_net_7x7',training_data_path)

########################################################################
# Train network and save network parameters
nn.train( training_image_set, annotation_type='Bodies', dilation_factor=-3,
                batch_size=10000, m_samples=200, n_batches=10, n_epochs=50)
# nn.save_network_parameters('centroid_net_5x5_2',training_data_path)
# nn.save_network_parameters('centroid_net_7x7',training_data_path)

########################################################################
# Display final performance
# print("\nTraining set performance:")
# nn.report_F1(training_image_set, annotation_type='Bodies',
#                 dilation_factor=-3, m_samples=5000, figure='Off')
#
# print("\nCross-validation set performance:")
# nn.report_F1(cv_image_set, annotation_type='Bodies',
#                 dilation_factor=-3, m_samples=5000, figure='Off')

# print("\nTesting F1 figure display:")
# nn.report_F1(small_training_image_set, annotation_type='Bodies',
#                 dilation_factor=-3, m_samples=2000, figure='On')

print('Done!\n')
