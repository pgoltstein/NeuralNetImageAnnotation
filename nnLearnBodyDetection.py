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
import os

#########################################################
# Arguments
parser = argparse.ArgumentParser( \
    description="Trains a deep convolutional neural network to detect " +\
                    "full cell bodies in an annotated image set")
parser.add_argument('name', type=str,
                    help= 'Name by which to identify the network')
parser.add_argument('-t', '--trainingdata', type=str,
                    help= 'Path to training data folder')
parser.add_argument('-n', '--networkpath', type=str,
                    help= 'Path to neural network folder')
parser.add_argument('-e', '--nepochs', type=int, default=100,
                    help='Number of epochs to train (default=100)')
parser.add_argument('-m', '--msamples', type=int, default=200,
                    help='Number of samples per training step (default=200)')
parser.add_argument('-s', '--size', type=int, default=36,
                    help='Size of the image annotations (default=36)')
parser.add_argument('-d', '--dropout', type=float, default=0.5,
                    help='Dropout fraction in fully connected " + \
                    "layer (default=36)')
parser.add_argument('-a', '--alpha', type=float, default=0.0005,
                    help="Learning rate 'alpha' (default=0.0005)")
parser.add_argument('-p', '--morph',  action="store_true",
                    help='Enables random morphing of annotations (default=off)')
parser.add_argument('-f', '--F1report', action="store_true",
                    help='Runs F1 report at the end of training (default=off)')
args = parser.parse_args()

# Set variables based on arguments
network_name = str(args.name)
n_epochs = args.nepochs
fc1_dropout = args.dropout
alpha = args.alpha
m_samples = args.msamples
annotation_size = (args.size,args.size)
morph_annotations = args.morph

########################################################################
# Settings and variables
if args.trainingdata:
    training_data_path = args.trainingdata
else:
    training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
if args.networkpath:
    network_path = args.networkpath
else:
    network_path = '/Users/pgoltstein/Dropbox/NeuralNets'

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
        network_path=os.path.join(network_path,network_name),
        input_image_size=annotation_size,
        n_input_channels=training_image_set.n_channels,
        output_size=(1,2),
        conv1_size=7, conv1_n_chan=32, conv1_n_pool=3,
        conv2_size=7, conv2_n_chan=64, conv2_n_pool=3,
        fc1_n_chan=256, fc1_dropout=fc1_dropout, alpha=alpha )

# Initialize and start
nn.start()

# Load network parameters
nn.restore()

# Display network architecture
nn.display_network_architecture()

########################################################################
# Train network
nn.train_epochs( training_image_set,
    annotation_type='Bodies',
    m_samples=m_samples, n_epochs=n_epochs, report_every=10,
    exclude_border=(40,40,40,40), morph_annotations=morph_annotations,
    rotation_list=rotation_list, scale_list_x=scale_list_x,
    scale_list_y=scale_list_y, noise_level_list=noise_level_list )

# Save network parameters and settings
nn.save()

########################################################################
# Display performance

if args.F1report:
    print("\nTraining set performance:")
    nn.report_F1( training_image_set,
        annotation_type='Bodies', m_samples=2000, morph_annotations=False,
        exclude_border=(40,40,40,40), show_figure='On')

    # Test morphed performance
    print("\nMorphed training set performance:")
    nn.report_F1( training_image_set, annotation_type='Bodies',
            m_samples=2000, exclude_border=(40,40,40,40),
            morph_annotations=True,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list,
            show_figure='On')

    plt.show()

print('Done!\n')
