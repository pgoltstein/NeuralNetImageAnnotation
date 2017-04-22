#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2017

Loads annotation data and trains a deep convolutional neural network to
detect whether the center pixel of a small zoom image is part of the
body or centroid of an annotation

@author: pgoltstein
"""

########################################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import ImageAnnotation as ia
import iaConvNetTools as cn
import argparse
import os

# Settings
import nnDefaultSettings as defaults

#########################################################
# Arguments
parser = argparse.ArgumentParser( \
    description="Trains a deep convolutional neural network with 2 " + \
            "convolutional layers and 1 fully connected layer to detect " + \
            "centroids, or full cell bodies, in an annotated image set. " + \
            "Runs on tensorflow framework. Learning is done using the " + \
            "ADAM optimizer. (written by Pieter Goltstein - April 2017)")

# Required arguments
parser.add_argument('annotationtype', type=str,
                    help= "'Centroids' or 'Bodies'")
parser.add_argument('name', type=str,
                    help= 'Name by which to identify the network')

# Annotation arguments
parser.add_argument('-z', '--size', type=int,
    default=defaults.annotation_size,
    help="Size of the image annotations (default={})".format(
        defaults.annotation_size))
parser.add_argument('-mp', '--morph',  action="store_true",
    default=defaults.morph_annotations,
    help='Flag enables random morphing of annotations (on/off; default={})'.format(
        "on" if defaults.morph_annotations else "off"))
parser.add_argument('-tpnr', '--includeannotationtypenr', type=int,
    default=defaults.include_annotation_typenr,
    help="Include only annotations of certain type_nr (default={})".format(
        defaults.include_annotation_typenr))
parser.add_argument('-dl', '--dilationfactor', type=int,
    default=defaults.dilation_factor,
    help="Dilation factor of annotations (default={})".format(
        defaults.dilation_factor))
parser.add_argument('-r', '--positivesampleratio', type=float,
    default=defaults.pos_sample_ratio,
    help='Ratio of positive vs negative samples (default={})'.format(
        defaults.pos_sample_ratio))
parser.add_argument('-ar', '--annotationborderratio', type=float,
    default=defaults.annotation_border_ratio,
    help='Ratio of samples from border between pos and neg samples' + \
        ' (default={})'.format(defaults.annotation_border_ratio))
parser.add_argument('-ch', '--imagechannels', nargs='+',
    default=defaults.use_channels,
    help="Select image channels to load (e.g. '-ch 1 2' " + \
        "loads first and second channel only; default=all)")
parser.add_argument('-nrm', '--normalizesamples',  action="store_true",
    default=defaults.normalize_samples,
    help="Normalizes the individual channels of the annotations " + \
        "(on/off; default={})".format("on" if defaults.normalize_samples else "off"))

# Training arguments
parser.add_argument('-tr', '--trainingprocedure', type=str,
    default=defaults.training_procedure,
    help= 'Training procedure (batch or epochs; default={})'.format(
        defaults.training_procedure))
parser.add_argument('-e', '--nepochs', type=int,
    default=defaults.n_epochs,
    help='Number of epochs to train (default={})'.format(
        defaults.n_epochs))
parser.add_argument('-m', '--msamples', type=int,
    default=defaults.m_samples,
    help='Number of samples per training step (default={})'.format(
        defaults.m_samples))
parser.add_argument('-nb', '--numberofbatches', type=int,
    default=defaults.number_of_batches,
    help='Number of training batches (default={})'.format(
        defaults.number_of_batches))
parser.add_argument('-b', '--batchsize', type=int,
    default=defaults.batch_size,
    help='Number of samples per training batch (default={})'.format(
        defaults.batch_size))
parser.add_argument('-rep', '--reportevery', type=int,
    default=defaults.report_every,
    help='Report every so many training epochs (default={})'.format(
        defaults.report_every))
parser.add_argument('-d', '--dropout', type=float,
    default=defaults.fc1_dropout,
    help='Dropout fraction in fully connected layer (default={})'.format(
        defaults.fc1_dropout))
parser.add_argument('-a', '--alpha', type=float,
    default=defaults.alpha,
    help="Learning rate 'alpha' (default={})".format(
        defaults.alpha))

# Network arguments
parser.add_argument('-net', '--nettype', type=str,
    default=defaults.network_type,
    help= "Type of network (1layer, 2layer, c2fc1; default={})".format(
        defaults.network_type))
parser.add_argument('-cz', '--convsize', type=int,
    default=defaults.conv_size,
    help="Size of convolutional filters (default={})".format(
        defaults.conv_size))
parser.add_argument('-cc', '--convchan', type=int,
    default=defaults.conv_chan,
    help="Number of convolutional filters (default={})".format(
        defaults.conv_chan))
parser.add_argument('-cp', '--convpool', type=int,
    default=defaults.conv_pool,
    help="Max pooling of convolutional filters (default={})".format(
        defaults.conv_pool))
parser.add_argument('-fz', '--fcsize', type=int,
    default=defaults.fc_size,
    help="Number of fully connected units (default={})".format(
        defaults.fc_size))

# Path arguments
parser.add_argument('-t', '--trainingdata', type=str,
    default=defaults.training_data_path,
    help= 'Path to training data folder (default={})'.format(
        defaults.training_data_path))
parser.add_argument('-n', '--networkpath', type=str,
    default=defaults.network_path,
    help= 'Path to neural network folder (default={})'.format(
        defaults.network_path))

# Output arguments
parser.add_argument('-lc', '--learningcurve', action="store_true",
    help='Displays the learning curve at the end of training (on/off default=off)')
parser.add_argument('-f1', '--F1report', action="store_true",
    help='Runs F1 report and displays examples of true/' + \
        'false positives/negatives at the end of training (on/off; default=off)')

# Parse arguments
args = parser.parse_args()

# Required arguments
network_name = args.name
annotation_type = args.annotationtype

# Annotation arguments
annotation_size = (args.size,args.size)
morph_annotations = args.morph
include_annotation_typenr = args.includeannotationtypenr
dilation_factor = args.dilationfactor
pos_sample_ratio = args.positivesampleratio
annotation_border_ratio = args.annotationborderratio
use_channels = args.imagechannels
normalize_samples = args.normalizesamples

# Training arguments
training_procedure = args.trainingprocedure
n_epochs = args.nepochs
m_samples = args.msamples
number_of_batches = args.numberofbatches
batch_size = args.batchsize
report_every = args.reportevery
fc1_dropout = args.dropout
alpha = args.alpha

# Network arguments
network_type = args.nettype
conv_size = args.convsize
conv_chan = args.convchan
conv_pool = args.convpool
fc_size = args.fcsize

# Path arguments
training_data_path = args.trainingdata
network_path = args.networkpath

########################################################################
# Other variables
exclude_border = defaults.exclude_border
normalize_images = defaults.normalize_images
rotation_list = np.arange( defaults.rotation_list[0],
                    defaults.rotation_list[1],defaults.rotation_list[2] )
scale_list_x = np.arange( defaults.scale_list_x[0],
                    defaults.scale_list_x[1],defaults.scale_list_x[2] )
scale_list_y = np.arange( defaults.scale_list_y[0],
                    defaults.scale_list_y[1],defaults.scale_list_y[2] )
noise_level_list = np.arange( defaults.noise_level_list[0],
                    defaults.noise_level_list[1],defaults.noise_level_list[2] )

########################################################################
# Load data
print("\nLoading data from directory into training_image_set:")
if use_channels is not None:
    for nr,ch in enumerate(use_channels):
        use_channels[nr] = int(ch)-1
training_image_set = ia.AnnotatedImageSet()
training_image_set.load_data_dir_tiff_mat( training_data_path,
    normalize=normalize_images, use_channels=use_channels )

########################################################################
# Set up network
if network_type.lower() == "1layer":
    nn = cn.NeuralNet1Layer( \
            network_path=os.path.join(network_path,network_name),
            input_image_size=annotation_size,
            n_input_channels=training_image_set.n_channels,
            output_size=(1,2), fc1_dropout=fc1_dropout, alpha=alpha )
elif network_type.lower() == "2layer":
    nn = cn.NeuralNet2Layer( \
            network_path=os.path.join(network_path,network_name),
            input_image_size=annotation_size,
            n_input_channels=training_image_set.n_channels,
            output_size=(1,2),
            fc1_n_chan=fc_size, fc1_dropout=fc1_dropout, alpha=alpha )
elif network_type.lower() == "c2fc1":
    nn = cn.ConvNetCnv2Fc1( \
            network_path=os.path.join(network_path,network_name),
            input_image_size=annotation_size,
            n_input_channels=training_image_set.n_channels,
            output_size=(1,2),
            conv1_size=conv_size, conv1_n_chan=conv_chan, conv1_n_pool=conv_pool,
            conv2_size=conv_size, conv2_n_chan=conv_chan*2, conv2_n_pool=conv_pool,
            fc1_n_chan=fc_size, fc1_dropout=fc1_dropout, alpha=alpha )

if nn.n_input_channels != training_image_set.n_channels:
    print("\n\nExisting network has been set up with {} input channels,\n \
        but function argument specified {} image channels.\n\n".format(
        nn.n_input_channels,training_image_set.n_channels) )
    print("Aborting network.\n")
    quit()

########################################################################
# Set training data
nn.log("\nUsing training_image_set from directory:")
nn.log(training_data_path)
nn.log(" >> " + training_image_set.__str__())
if use_channels is None:
    nn.log("Using all available {} image channels".format(
            training_image_set.n_channels))
else:
    nn.log("Using image channels {} (zero-based)".format(use_channels))

if annotation_type == 'Centroids':
    if dilation_factor == -999:
        dilation_factor = 3
    nn.log("Setting centroid dilation factor of the image " + \
                                    "to {}".format(dilation_factor))
    training_image_set.centroid_dilation_factor = dilation_factor

elif annotation_type == 'Bodies':
    if dilation_factor == -999:
        dilation_factor = 0
    nn.log("Setting body dilation factor of the image " + \
                                    "to {}".format(dilation_factor))
    training_image_set.body_dilation_factor = dilation_factor

if include_annotation_typenr is not None:
    nn.log("Setting annotation typenr to be included " + \
                                    "to {}".format(include_annotation_typenr))
    training_image_set.include_annotation_typenr = include_annotation_typenr

########################################################################
# Initialize and start
nn.start()

# Load network parameters
nn.restore()

# Display network architecture
nn.display_network_architecture()

########################################################################
# Train network
if training_procedure.lower() == "epochs":
    nn.train_epochs( training_image_set,
        annotation_type=annotation_type,
        m_samples=m_samples, n_epochs=n_epochs, report_every=report_every,
        exclude_border=exclude_border, pos_sample_ratio=pos_sample_ratio,
        annotation_border_ratio=annotation_border_ratio,
        normalize_samples=normalize_samples, morph_annotations=morph_annotations,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list )
elif training_procedure.lower() == "batch":
    nn.train_batch( training_image_set, n_batches=number_of_batches,
        batch_size=batch_size, m_samples=m_samples, n_epochs=n_epochs,
        annotation_type=annotation_type,
        exclude_border=exclude_border, pos_sample_ratio=pos_sample_ratio,
        annotation_border_ratio=annotation_border_ratio,
        normalize_samples=normalize_samples, morph_annotations=morph_annotations,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list )

# Save network parameters and settings
nn.save()

########################################################################
# Display performance

if args.learningcurve:
    nn.log("\nDisplay learning curve and filters:")
    nn.show_learning_curve()
    nn.show_filters()

if args.F1report:
    nn.log("\nTraining set performance:")
    nn.report_F1( training_image_set,
        annotation_type=annotation_type, m_samples=2000, pos_sample_ratio=pos_sample_ratio,
        annotation_border_ratio=annotation_border_ratio,
        morph_annotations=False, exclude_border=(40,40,40,40),
        channel_order=None, show_figure='On')

    # Test morphed performance
    nn.log("\nMorphed training set performance:")
    nn.report_F1( training_image_set, annotation_type=annotation_type,
            m_samples=2000, exclude_border=(40,40,40,40), pos_sample_ratio=0.5,
            morph_annotations=True,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list,
            channel_order=None, show_figure='On')

if args.learningcurve or args.F1report:
    plt.show()

nn.log('Done!\n')
