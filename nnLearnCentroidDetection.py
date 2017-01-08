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
test_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet2'
zoom_size = (32,32)
training_image_set = ia.AnnotatedImageSet()
training_image_set.load_data_dir(training_data_path)
test_image_set = ia.AnnotatedImageSet()
test_image_set.load_data_dir(test_data_path)

########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1( input_image_size=zoom_size, n_input_channels=3,
                        output_size=(1,2), alpha=1e-4 )
nn.start()
# nn.load_network_parameters('centroid_net',training_data_path)

########################################################################
# Train network and save network parameters
nn.train( training_image_set, m_samples=500, n_epochs=1000, display_every_n=25)
nn.save_network_parameters('centroid_net',training_data_path)

########################################################################
# Display final performance
print("\nTraining data performance:")
nn.report_F1(training_image_set, 5000)

print("\nTest data performance:")
nn.report_F1(test_image_set, 5000)

print('Done!\n')
