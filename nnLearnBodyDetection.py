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
parser.add_argument('-z', '--size', type=int, default=27,
                    help='Size of the image annotations (default=27)')

parser.add_argument('-t', '--trainingdata', type=str,
                    help= 'Path to training data folder')
parser.add_argument('-n', '--networkpath', type=str,
                    help= 'Path to neural network folder')

parser.add_argument('-e', '--nepochs', type=int, default=100,
                    help='Number of epochs to train (default=100)')
parser.add_argument('-m', '--msamples', type=int, default=200,
                    help='Number of samples per training step (default=200)')

parser.add_argument('-d', '--dropout', type=float, default=0.5,
                    help='Dropout fraction in fully connected " + \
                    "layer (default=0.5)')
parser.add_argument('-a', '--alpha', type=float, default=0.0004,
                    help="Learning rate 'alpha' (default=0.0004)")

parser.add_argument('-cz', '--convsize', type=int, default=5,
                    help="Size of convolutional filters (default=5)")
parser.add_argument('-cc', '--convchan', type=int, default=32,
                    help="Number of convolutional filters (default=32)")
parser.add_argument('-cp', '--convpool', type=int, default=2,
                    help="Max pooling of convolutional filters (default=2)")
parser.add_argument('-fz', '--fcsize', type=int, default=256,
                    help="Number of fully connected units (default=256)")

parser.add_argument('-mp', '--morph',  action="store_true",
                    help='Enables random morphing of annotations (default=off)')
parser.add_argument('-f1', '--F1report', action="store_true",
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
conv_size = args.convsize
conv_chan = args.convchan
conv_pool = args.convpool
fc_size = args.fcsize

if args.trainingdata:
    training_data_path = args.trainingdata
else:
    training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
if args.networkpath:
    network_path = args.networkpath
else:
    network_path = '/Users/pgoltstein/Dropbox/NeuralNets'

########################################################################
# Settings and variables
rotation_list = np.array(range(360))
scale_list_x = np.array(range(900,1100)) / 1000
scale_list_y = np.array(range(900,1100)) / 1000
noise_level_list = np.array(range(200)) / 10000

########################################################################
# Load data
print("\nLoading data from directory into training_image_set:")
training_image_set = ia.AnnotatedImageSet()
training_image_set.load_data_dir_tiff_mat(training_data_path)

########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1( \
        network_path=os.path.join(network_path,network_name),
        input_image_size=annotation_size,
        n_input_channels=training_image_set.n_channels,
        output_size=(1,2),
        conv1_size=conv_size, conv1_n_chan=conv_chan, conv1_n_pool=conv_pool,
        conv2_size=conv_size, conv2_n_chan=conv_chan*2, conv2_n_pool=conv_pool,
        fc1_n_chan=fc_size, fc1_dropout=fc1_dropout, alpha=alpha )

nn.log("\nUsing training_image_set from directory:")
nn.log(training_data_path)
nn.log(" >> " + training_image_set.__str__())

########################################################################
# Dilate bodies
nn.log("Setting body dilation factor of the image set to 0")
training_image_set.body_dilation_factor = 0

########################################################################
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
    nn.log("\nDisplay learning curve:")
    nn.show_learning_curve()

    nn.log("\nTraining set performance:")
    nn.report_F1( training_image_set,
        annotation_type='Bodies', m_samples=2000, morph_annotations=False,
        exclude_border=(40,40,40,40), show_figure='On')

    # Test morphed performance
    nn.log("\nMorphed training set performance:")
    nn.report_F1( training_image_set, annotation_type='Bodies',
            m_samples=2000, exclude_border=(40,40,40,40),
            morph_annotations=True,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list,
            show_figure='On')

    plt.show()

nn.log('Done!\n')
