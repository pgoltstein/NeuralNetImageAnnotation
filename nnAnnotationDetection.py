#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2017

Contains functions that detects centroids or bodies of annotations

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import ImageAnnotation as ia
import iaConvNetTools as cn
import argparse
import os

from scipy import ndimage
from skimage import measure


#########################################################
# Arguments
parser = argparse.ArgumentParser( \
    description="Detects centroids, or full cell bodies, using a trained " + \
            " deep convolutional neural network with 2 convolutional " + \
            "layers and 1 fully connected layer. Saves as annotated image." + \
            "Runs on tensorflow framework. " + \
            "(written by Pieter Goltstein - April 2017)")

parser.add_argument('networkname', type=str,
                    help= 'Name by which to identify the network for ' +\
                            'detecting cell bodies')
parser.add_argument('image', type=str,
                    help= 'Filename (and path) of image to classify')

parser.add_argument('-c', '--centroidnet', type=str,
                    help= 'Name by which to identify the network for ' +\
                            'detecting cell centroids')
parser.add_argument('-bx', '--borderexclude', type=str,
                    help= 'Name by which to identify the file holding ' +\
                            'border exclusion parameters (matlab saved ' +\
                            'parameters "left")')
parser.add_argument('-n', '--networkpath', type=str,
                    help= 'Path to neural network folder')
parser.add_argument('-ch', '--imagechannels', nargs='+',
                    help="Select image channels to load (e.g. '-ch 1 2' " + \
                    "loads first and second channel only; default=all)")
args = parser.parse_args()

# Set variables based on arguments
network_name = str(args.networkname)
annotation_type = str(args.annotationtype)
image_name = str(args.imagename)
use_centroidnet = args.centroidnet
use_channels = args.imagechannels
if args.networkpath:
    network_path = args.networkpath
else:
    network_path = '/Users/pgoltstein/Dropbox/NeuralNets'
if args.imagedata:
    image_path = args.imagedata
else:
    image_path = '.'

########################################################################
# Other variables
normalize_images = True

########################################################################
# Load data
print("\nLoading {} into AnnotatedImage class".format(image_name))
if use_channels is not None:
    for nr,ch in enumerate(use_channels):
        use_channels[nr] = int(ch)-1
anim = ia.AnnotatedImage()
anim.add_image_from_file(file_name=image_name,file_path=image_path,
                normalize=normalize_images, use_channels=use_channels)

# Border exclusion (movement artifact)
if args.borderexclude is not None:
    anim.exclude_border=args.borderexclude

print(" >> " + anim.__str__())

########################################################################
# If requested, set up network for detecting centroids
if use_centroidnet:
    nn_ctr = cn.ConvNetCnv2Fc1( logging=False, \
            network_path=os.path.join(network_path,use_centroidnet) )
    if nn_ctr.n_input_channels != anim.n_channels:
        print("\n\nExisting network has been set up with {} input channels,\n \
            but function argument specified {} image channels.\n\n".format(
            nn_ctr.n_input_channels,anim.n_channels) )
        print("Aborting network.\n")
        quit()
    if use_channels is None:
        nn_ctr.log("Using all available {} image channels".format(
                anim.n_channels))
    else:
        nn_ctr.log("Using image channels {} (zero-based)".format(use_channels))
    annotation_size = (nn_ctr.y_res,nn_ctr.x_res)

    # Initialize and start, load network parameters, display network architecture
    nn_ctr.start()
    nn_ctr.restore()
    nn_ctr.display_network_architecture()

    # Annotate centroids
    nn_ctr.log("Running centroid detection:")
    anim.detected_centroids = nn_ctr.annotate_image( anim )

########################################################################
# Set up network for detecting bodies
nn = cn.ConvNetCnv2Fc1( logging=False, \
        network_path=os.path.join(network_path,network_name) )
if nn.n_input_channels != anim.n_channels:
    print("\n\nExisting network has been set up with {} input channels,\n \
        but function argument specified {} image channels.\n\n".format(
        nn.n_input_channels,anim.n_channels) )
    print("Aborting network.\n")
    quit()
if use_channels is None:
    nn.log("Using all available {} image channels".format(
            anim.n_channels))
else:
    nn.log("Using image channels {} (zero-based)".format(use_channels))
annotation_size = (nn.y_res,nn.x_res)

# Initialize and start, load network parameters, display network architecture
nn.start()
nn.restore()
nn.display_network_architecture()

# Annotate image
nn_ctr.log("Running body detection:")
anim.detected_bodies = nn.annotate_image( anim )

########################################################################
# Convert classified images to annotations
anim.detected_bodies = classified_bodies



########################################################################
# Save classified image
np.save( os.path.join( image_path,
    'classified_image'+annotation_type), classified_image)
print( "Saved classified_image as: {}".format( os.path.join( image_path,
    'classified_image'+annotation_type) ) )

# anim.detected_bodies = np.load( os.path.join( image_path,
#     'classified_imageBodies.npy') )
# anim.detected_centroids = np.load( os.path.join( image_path,
#     'classified_imageCentroids.npy') )
# anim.exclude_border=(0,20,10,3)
#
# anim.generate_cnn_annotations_cb( min_size=120, max_size=None,
#     dilation_factor_centroids=-1, dilation_factor_bodies=-1 )
#
# RGB = np.zeros((anim.detected_bodies.shape[0],anim.detected_bodies.shape[1],3))
# RGB[:,:,1] = anim.detected_centroids
# RGB[:,:,2] = anim.detected_bodies
#
#
# # ************************************************************
# # Show matplotlib images
#
# # Show image and classification result
# with sns.axes_style("white"):
#     plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
#     axr = plt.subplot2grid( (1,2), (0,0) )
#     axr.imshow( anim.RGB(),
#         interpolation='nearest', vmax=anim.RGB().max()*0.7)
#     axr.set_title("Image")
#     plt.axis('tight')
#     plt.axis('off')
#
#     axb = plt.subplot2grid( (1,2), (0,1) )
#     axb.imshow(RGB)
#     axb.set_title("Annotated bodies and centroids")
#     plt.axis('tight')
#     plt.axis('off')
#
# with sns.axes_style("white"):
#     plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
#     axr = plt.subplot2grid( (1,2), (0,0) )
#     axr.imshow( anim.RGB(),
#         interpolation='nearest', vmax=anim.RGB().max()*0.7)
#     for an in anim.annotation:
#         axr.plot( an.perimeter[:,1], an.perimeter[:,0],
#             linewidth=1, color="#ffffff" )
#     axr.set_title("Annotated image")
#     plt.axis('tight')
#     plt.axis('off')
#
#     axr = plt.subplot2grid( (1,2), (0,1) )
#     axr.imshow(RGB)
#     for an in anim.annotation:
#         axr.plot( an.perimeter[:,1], an.perimeter[:,0],
#             linewidth=1, color="#ffffff" )
#     axr.set_title("Annotated image")
#     plt.axis('tight')
#     plt.axis('off')






# # Show image and classification result
# with sns.axes_style("white"):
#     plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
#     ax = list(range(3))
#     for ch in range(3):
#         ax[ch] = plt.subplot2grid( (2,3), (0,ch) )
#         ax[ch].imshow( anim.channel[ch], cmap='gray',
#             interpolation='nearest', vmax=anim2.channel[ch].max()*0.7 )
#         ax[ch].set_title("anim2: Channel{}".format(ch))
#         plt.axis('tight')
#         plt.axis('off')
#
#     axb = plt.subplot2grid( (2,3), (1,0) )
#     axb.imshow(anim1.bodies>0,vmin=-0.1)
#     axb.set_title("anim: Annotated bodies")
#     plt.axis('tight')
#     plt.axis('off')
#
#     axb = plt.subplot2grid( (2,3), (1,1) )
#     axb.imshow(classified_image,vmin=-0.1)
#     axb.set_title("anim2: Classification result")
#     plt.axis('tight')
#     plt.axis('off')
#
#     axr = plt.subplot2grid( (2,3), (1,2) )
#     axr.imshow( anim1.RGB(),
#         interpolation='nearest', vmax=anim1.RGB().max()*0.7)
#     for an in anim1.annotation:
#         axr.plot( an.perimeter[:,1], an.perimeter[:,0],
#             linewidth=1, color="#ffffff" )
#     axr.set_title("anim: RGB with annotations")
#     plt.axis('tight')
#     plt.axis('off')

print('Done!\n')
plt.show()
