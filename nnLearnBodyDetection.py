#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2017

Contains functions that detect bodies of annotations in a single step

@author: pgoltstein
"""

########################################################################
### Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ImageAnnotation as ia
import iaConvNetTools as cn
import argparse

#########################################################
# Arguments
parser = argparse.ArgumentParser( \
    description="Trains a deep convolutional neural network to detect " +\
                    "cell bodies in an annotated image set")
parser.add_argument('nEpochs', help='Number of epochs to train')
args = parser.parse_args()
n_epochs = int(args.nEpochs)

########################################################################
# Settings and variables
annotation_size = (27,27)
training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
network_path = '/Users/pgoltstein/Dropbox/TEMP'
network_name = 'nnBodyDetect'
rotation_list = np.array(range(360))
scale_list_x = np.array(range(900,1100)) / 1000
scale_list_y = np.array(range(900,1100)) / 1000
noise_level_list = np.array(range(200)) / 10000

########################################################################
# Load data
print("\nLoading data from directory into training_image_set:")
print(training_data_path)
training_image_set = ia.AnnotatedImageSet()
training_image_set.load_data_dir_tiff_mat(training_data_path)
print(" >> " + training_image_set.__str__())

# Dilate bodies
print("Setting body dilation factor of the image set to 0")
training_image_set.body_dilation_factor = 0

########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1( \
        input_image_size=annotation_size,
        n_input_channels=training_image_set.n_channels, output_size=(1,2),
        conv1_size=7, conv1_n_chan=32, conv1_n_pool=3,
        conv2_size=7, conv2_n_chan=64, conv2_n_pool=3,
        fc1_n_chan=256, fc1_dropout=0.5, alpha=4e-4 )
nn.start()

# Load network parameters
# nn.load_network_parameters(network_name,network_path)

########################################################################
# Train network and save network parameters
print("Training network for {} epochs".format(n_epochs))
nn.train_epochs( training_image_set,
    annotation_type='Bodies', m_samples=200, n_epochs=n_epochs, report_every=10,
    exclude_border=(40,40,40,40), morph_annotations=True,
    rotation_list=rotation_list, scale_list_x=scale_list_x,
    scale_list_y=scale_list_y, noise_level_list=noise_level_list )

# Save network parameters
nn.save_network_parameters(network_name,network_path)

########################################################################
# Display performance
print("\nTraining set performance:")
nn.report_F1( training_image_set,
    annotation_type='Bodies', m_samples=2000, morph_annotations=False,
    exclude_border=(40,40,40,40), show_figure='On')

########################################################################
# Test morphed performance
print("\nMorphed training set performance:")
nn.report_F1( training_image_set, annotation_type='Bodies',
        m_samples=2000, exclude_border=(40,40,40,40), morph_annotations=True,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list,
        show_figure='On')

print('Done!\n')
plt.show()
