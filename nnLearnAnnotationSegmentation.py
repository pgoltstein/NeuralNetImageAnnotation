#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 27, 2017

Loads annotation data and trains a deep convolutional neural network to
annotate all pixels in a small zoom image that are part of the
body that the zoom image is centered on

@author: pgoltstein
"""

########################################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import ImageAnnotation as ia
import iaConvNetSegmentation as cn
import argparse
import os

# Settings
import nnSegmentationDefaults as defaults

#########################################################
# Arguments
parser = argparse.ArgumentParser( \
    description="Trains a deep convolutional neural network with 2 " + \
            "convolutional layers and 1 fully connected layer to detect " + \
            "which pixels are part of the annotation in the center. " + \
            "Runs on tensorflow framework. Learning is done using the " + \
            "ADAM optimizer. (written by Pieter Goltstein - April 2017)")

# Required arguments
parser.add_argument('annotationtype', type=str,
                    help= "To be annotated region: 'Centroids' or 'Bodies'")
parser.add_argument('name', type=str,
                    help= 'Name by which to identify the network')

# Annotation arguments
parser.add_argument('-stp', '--selectiontype', type=str,
    default=defaults.selection_type,
    help= "Select positive annotations from ('Centroids' or 'Bodies'; " + \
        "default={})".format(defaults.selection_type))
parser.add_argument('-all', '--segmentall',  action="store_true",
    default=defaults.segment_all,
    help='Flag enables segmantation of all (instead of single) annotations ' + \
        ' (on/off; default={})'.format("on" if defaults.segment_all else "off"))
parser.add_argument('-iz', '--imagesize', type=int,
    default=defaults.image_size,
    help="Size of the input images (default={})".format(
        defaults.image_size))
parser.add_argument('-az', '--annotationsize', type=int,
    default=defaults.annotation_size,
    help="Size of the image annotations (default={})".format(
        defaults.annotation_size))
parser.add_argument('-mp', '--morph',  action="store_true",
    default=defaults.morph_annotations,
    help='Flag enables random morphing of annotations (on/off; default={})'.format(
        "on" if defaults.morph_annotations else "off"))
parser.add_argument('-tpnr', '--includeannotationtypenr', type=int, default=1,
    help="Include only annotations of certain type_nr (default=1)")
parser.add_argument('-dlc', '--centroiddilationfactor', type=int,
    default=defaults.centroid_dilation_factor,
    help="Dilation factor of annotation centers (default={})".format(
        defaults.centroid_dilation_factor))
parser.add_argument('-dlb', '--bodydilationfactor', type=int,
    default=defaults.body_dilation_factor,
    help="Dilation factor of the to be annotated bodies (default={})".format(
        defaults.body_dilation_factor))
parser.add_argument('-r', '--positivesampleratio', type=float,
    default=defaults.sample_ratio,
    help='Ratio of positive vs negative samples (default={})'.format(
        defaults.sample_ratio))
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
parser.add_argument('-dim', '--downsampleimage', type=float,
    default=defaults.downsample_image,
    help='Downsampling factor for the annotated images, borders and annotations' + \
        ' (default={})'.format(defaults.downsample_image))
parser.add_argument('-brdr', '--bordermargin', type=int, default=None,
    help="Margin to stay away from image borders (default=None)")

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
    default=defaults.fc_dropout,
    help='Dropout fraction in fully connected layer (default={})'.format(
        defaults.fc_dropout))
parser.add_argument('-a', '--alpha', type=float,
    default=defaults.alpha,
    help="Learning rate 'alpha' (default={})".format(
        defaults.alpha))

# Network arguments
parser.add_argument('-net', '--nettype', type=str,
    default=defaults.network_type,
    help= "Type of network (1layer, 2layer, c2fc1, cNfc1; default={})".format(
        defaults.network_type))
parser.add_argument('-clay', '--convnlayers', type=int,
    default=defaults.conv_n_layers,
    help="Number of convolutional layers (default={})".format(
        defaults.conv_n_layers))
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
parser.add_argument('-f1', '--F1report', type=str, default=None,
    help='Runs F1 report and displays examples of true/' + \
        'false positives/negatives at the end of training ' + \
        'takes path with (training/cv/test) annotated images as argument')

# Parse arguments
args = parser.parse_args()

# Required arguments
network_name = args.name
annotation_type = args.annotationtype

# Annotation arguments
selection_type = args.selectiontype
segment_all = args.segmentall
image_size = (args.imagesize,args.imagesize)
annotation_size = (args.annotationsize,args.annotationsize)
morph_annotations = args.morph
include_annotation_typenrs = args.includeannotationtypenr
centroid_dilation_factor = args.centroiddilationfactor
body_dilation_factor = args.bodydilationfactor
sample_ratio = args.positivesampleratio
annotation_border_ratio = args.annotationborderratio
use_channels = args.imagechannels
normalize_samples = args.normalizesamples
downsample_image = args.downsampleimage

# Training arguments
training_procedure = args.trainingprocedure
n_epochs = args.nepochs
m_samples = args.msamples
number_of_batches = args.numberofbatches
batch_size = args.batchsize
report_every = args.reportevery
fc_dropout = args.dropout
alpha = args.alpha

# Network arguments
network_type = args.nettype
conv_n_layers = args.convnlayers
conv_size = args.convsize
conv_chan = args.convchan
conv_pool = args.convpool
fc_size = args.fcsize

# Path arguments
training_data_path = args.trainingdata
network_path = args.networkpath

########################################################################
# Other variables
if ~isinstance(args.bordermargin,int):
    exclude_border = defaults.exclude_border
else:
    exclude_border = [args.bordermargin,args.bordermargin,args.bordermargin,args.bordermargin]
normalize_images = defaults.normalize_images
rotation_list = np.arange( defaults.rotation_list[0],
                    defaults.rotation_list[1],defaults.rotation_list[2] )
scale_list_x = np.arange( defaults.scale_list_x[0],
                    defaults.scale_list_x[1],defaults.scale_list_x[2] )
scale_list_y = np.arange( defaults.scale_list_y[0],
                    defaults.scale_list_y[1],defaults.scale_list_y[2] )
noise_level_list = np.arange( defaults.noise_level_list[0],
                    defaults.noise_level_list[1],defaults.noise_level_list[2] )

# If no training, skip loading data
perform_network_training = True
if (training_procedure.lower()=="epochs" and n_epochs==0) or \
    (training_procedure.lower()=="batch" and number_of_batches==0):
    perform_network_training = False

########################################################################
# Load data
if use_channels is not None:
    for nr,ch in enumerate(use_channels):
        use_channels[nr] = int(ch)-1

if perform_network_training:
    print("\nLoading data from directory into training_image_set:")
    training_image_set = ia.AnnotatedImageSet(downsample=downsample_image)
    training_image_set.include_annotation_typenrs = include_annotation_typenrs
    training_image_set.load_data_dir_tiff_mat( training_data_path,
        normalize=normalize_images, use_channels=use_channels,
        exclude_border=exclude_border )
    n_input_channels = training_image_set.n_channels
    print(training_image_set.ai_list[0].__str__())
else:
    n_input_channels = None


########################################################################
# Set up network
if network_type.lower() == "cnfc1":
    nn = cn.ConvNetCnv2Fc1Nout( \
            network_path=os.path.join(network_path,network_name),
            input_image_size=image_size,
            n_input_channels=n_input_channels,
            output_image_size=annotation_size,
            conv_n_layers = conv_n_layers,
            conv_size=conv_size, conv_n_chan=conv_chan, conv_n_pool=conv_pool,
            fc1_n_chan=fc_size, fc_dropout=fc_dropout, alpha=alpha )
if network_type.lower() == "c1fc2":
    nn = cn.ConvNetCnv1Fc2Nout( \
            network_path=os.path.join(network_path,network_name),
            input_image_size=image_size,
            n_input_channels=n_input_channels,
            output_image_size=annotation_size,
            conv1_size=conv_size, conv1_n_chan=conv_chan, conv1_n_pool=conv_pool,
            fc1_n_chan=fc_size, fc2_n_chan=fc_size,
            fc_dropout=fc_dropout, alpha=alpha )


########################################################################
# Set training data
if perform_network_training:
    nn.log("\nUsing training_image_set from directory:")
    nn.log(training_data_path)
    nn.log(" >> " + training_image_set.__str__())
    if use_channels is None:
        nn.log("Using all available {} image channels".format(
                training_image_set.n_channels))
    else:
        nn.log("Using image channels {} (zero-based)".format(use_channels))

    if annotation_type.lower() == 'centroids' or selection_type.lower() == 'centroids':
        nn.log("Setting centroid dilation factor of the image " + \
                                        "to {}".format(centroid_dilation_factor))
        training_image_set.centroid_dilation_factor = centroid_dilation_factor

    elif annotation_type.lower() == 'bodies' or selection_type.lower() == 'bodies':
        nn.log("Setting body dilation factor of the image " + \
                                        "to {}".format(body_dilation_factor))
        training_image_set.body_dilation_factor = body_dilation_factor

    nn.log("Training on annotation class: {}".format( \
        training_image_set.class_labels))

########################################################################
# Initialize and start
nn.start()

# Load network parameters
nn.restore()

# Display network architecture
nn.display_network_architecture()

########################################################################
# Train network
if perform_network_training:
    if training_procedure.lower() == "epochs":
        nn.train_epochs( training_image_set, m_samples=m_samples,
            n_epochs=n_epochs, report_every=report_every,
            selection_type=selection_type, annotation_type=annotation_type,
            annotation_border_ratio=annotation_border_ratio,
            sample_ratio=sample_ratio,
            normalize_samples=normalize_samples, segment_all=segment_all,
            morph_annotations=morph_annotations,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list )
    elif training_procedure.lower() == "batch":
        nn.train_batch( training_image_set, n_batches=number_of_batches,
            batch_size=batch_size, m_samples=m_samples, n_epochs=n_epochs,
            selection_type=selection_type, annotation_type=annotation_type,
            annotation_border_ratio=annotation_border_ratio,
            sample_ratio=sample_ratio,
            normalize_samples=normalize_samples, segment_all=segment_all,
            morph_annotations=morph_annotations,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list )

    # Save network parameters and settings
    nn.save()

########################################################################
# Display learning curve and filters

if args.learningcurve:
    nn.log("\nDisplay learning curve and filters:")
    nn.show_learning_curve()
    nn.show_filters()

########################################################################
# Generate F1 report for training/cv/test data

if args.F1report is not None:
    nn.log("\nGenerating F1 report")
    if args.F1report.lower() == "same" and perform_network_training:
        nn.log("Using training data for F1 report")
        f1_path = training_data_path
        f1_image_set = training_image_set
        nn.log(" >> " + f1_image_set.__str__())
    else:
        if args.F1report.lower() == "same":
            f1_path = training_data_path
        else:
            f1_path = args.F1report
        nn.log("Loading data from {}:".format(f1_path))
        f1_image_set = ia.AnnotatedImageSet(downsample=downsample_image)
        f1_image_set.include_annotation_typenrs = include_annotation_typenrs
        f1_image_set.load_data_dir_tiff_mat( f1_path,
            normalize=normalize_images, use_channels=use_channels,
            exclude_border=exclude_border )
        nn.log(" >> " + f1_image_set.__str__())
        if annotation_type.lower() == 'centroids' or selection_type.lower() == 'centroids':
            nn.log("Setting centroid dilation factor of the image " + \
                                            "to {}".format(centroid_dilation_factor))
            f1_image_set.centroid_dilation_factor = centroid_dilation_factor
        elif annotation_type.lower() == 'bodies' or selection_type.lower() == 'bodies':
            nn.log("Setting body dilation factor of the image " + \
                                            "to {}".format(body_dilation_factor))
            f1_image_set.body_dilation_factor = body_dilation_factor
        nn.log("Testing on annotation class: {}".format(f1_image_set.class_labels))

    # Test morphed performance
    nn.log("\nPerformance of {}:".format(f1_path))
    nn.report_F1( f1_image_set,
            selection_type=selection_type, annotation_type=annotation_type,
            m_samples=1000,
            annotation_border_ratio=annotation_border_ratio,
            sample_ratio=[0,1],
            normalize_samples=normalize_samples, segment_all=segment_all,
            morph_annotations=False,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list,
            channel_order=None, show_figure='On')

if args.learningcurve or (args.F1report is not None):
    plt.show()

nn.log('Done!\n')
