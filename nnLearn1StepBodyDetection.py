#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2017

Contains functions that detect bodies of annotations in a single step

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
annotation_size = (36,36)
training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
network_path = '/Users/pgoltstein/Dropbox/TEMP'
print("\nLoading data from directory into training_image_set:")
print(training_data_path)
training_image_set = ia.AnnotatedImageSet()
training_image_set.load_data_dir_tiff_mat(training_data_path)
print(" >> " + training_image_set.__str__())

# Dilate centroids
print("Changing body dilation factor of the image set to -3")
training_image_set.body_dilation_factor = -3

########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1( \
        input_image_size=annotation_size,
        n_input_channels=training_image_set.n_channels, output_size=(1,2),
        conv1_size=5, conv1_n_chan=16, conv1_n_pool=2,
        conv2_size=5, conv2_n_chan=32, conv2_n_pool=2,
        fc1_n_chan=256, fc1_dropout=0.5, alpha=4e-4 )
nn.start()
# nn.load_network_parameters('body1step_net',network_path)

########################################################################
# Train network and save network parameters
nn.train_epochs( training_image_set,
    annotation_type='Bodies', m_samples=500, n_epochs=2500,
    exclude_border=(40,40,40,40))
nn.save_network_parameters('body1step_net',network_path)

########################################################################
# Display performance
print("\nTraining set performance:")
nn.report_F1( training_image_set,
    annotation_type='Bodies', m_samples=2000,
    exclude_border=(40,40,40,40), show_figure='On')

########################################################################
# Test morphed performance
print("\nMorphed training set performance:")
rotation_list = np.array(range(360))
scale_list_x = np.array(range(900,1100)) / 1000
scale_list_y = np.array(range(900,1100)) / 1000
noise_level_list = np.array(range(200)) / 10000
nn.report_F1( training_image_set, annotation_type='Bodies',
        m_samples=2000, exclude_border=(40,40,40,40), morph_annotations=True,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list,
        show_figure='On')

print('Done!\n')
