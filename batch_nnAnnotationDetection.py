#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 09 2017

Batch loops through a directory structure and creates ROIs for all
'channel.mat' files

@author: pgoltstein
"""

### Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ImageAnnotation as ia
import iaConvNetSingleOutput as cn
import argparse
import os
import glob
from scipy import ndimage
from skimage import measure

# Arguments
parser = argparse.ArgumentParser( \
    description = \
        "Batch-detects annotations by classifying pixels as being part of " + \
        "full cell bodies (and optional centroids), using a trained " + \
        " deep convolutional neural network with N convolutional " + \
        "layers and 1 fully connected layer. Saves as annotated image." + \
        "Runs on tensorflow framework. " + \
        "(written by Pieter Goltstein - August 2017)")

parser.add_argument('datapath', type=str,
    help= 'Path with subfolders that contain Exp# (blabla) directories')
parser.add_argument('-a', '--detectfromanim', action="store_true",
    help='Loads annotations of already saved anim.npy files (on/off default=off)')
args = parser.parse_args()

# Settings
data_path = str(args.datapath)
n_multilevel_layers = 2
body_net = "D:/neuralnets/bnet002"
centroid_net = "D:/neuralnets/cnet002"
use_channels = [0,1,3] # zero based
min_size = 120
max_size = 750
dilation_factor_centroids = 0
dilation_factor_bodies = 0
re_dilate_bodies = 0
normalize_images = True


def load_anim( im_file, bdr_file ):
    """ Load channels.mat and border.mat files into an instance of
    the annotated image class
    """
    anim = ia.AnnotatedImage()

    # Find and load image file
    print("Loading image: {}".format(im_file))
    anim.add_image_from_file(file_name=im_file,file_path='',
                    normalize=normalize_images, use_channels=use_channels)

    # Find and load border exclusion file
    print("Loading border exclusion: {}".format(bdr_file))
    anim.exclude_border = bdr_file

    # Display and return
    print(" >> " + anim.__str__())
    return anim


def nn_annotate_anim( anim, body_net, centroid_net ):
    """ Annotates the centroids and bodies in the supplied instance of
    the annotated image class
    """

    # Load centroid net
    nn_ctr = cn.ConvNetCnvNFc1( logging=False,
        network_path=centroid_net, reduced_output=True )
    nn_ctr.start()
    nn_ctr.restore()

    # Annotate centroids
    print("Running centroid detection")
    anim.detected_centroids = nn_ctr.annotate_image( anim )
    nn_ctr.close()

    # Load body net
    nn_bdy = cn.ConvNetCnvNFc1( logging=False,
        network_path=body_net, reduced_output=True )
    nn_bdy.start()
    nn_bdy.restore()

    # Annotate image
    print("Running body detection")
    anim.detected_bodies = nn_bdy.annotate_image( anim )
    nn_bdy.close()

def save_annotations(anim, filepath, layer_no):
    """ Saves the annotated image and exports the ROIs
    """

    # Concatenate filename, path and layer_no
    filebase = os.path.join( filepath, 'nnAnIm' + "-L{}".format(layer_no) )
    ROIbase = os.path.join( filepath, "nnROI{}".format(layer_no) )

    # Add version counter to prevent overwriting
    version_cnt = 0
    while glob.glob(filebase+"-v{:d}".format(version_cnt)+".npy"):
        version_cnt += 1
    anim_name = filebase+"-v{:d}".format(version_cnt)

    # Save annotated image
    anim.save(file_name=anim_name, file_path='')
    anim.export_annotations_to_mat( file_name=ROIbase, file_path='')

def get_exp_folders(data_path):
    """ Detects all folders with Exp in the name, and returns sorted list
    """
    exp_folders = []
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        if "Exp" in dirpath:
            exp_folders.append(dirpath)
    return sorted(exp_folders)


########################################################################
# Main loop across all folders; locations, dates, experiments
for exp_path in get_exp_folders(data_path):

        # Loop multilevel layers
        for layer_no in range(1,n_multilevel_layers+1):

            if args.detectfromanim:
                anim_files = sorted(glob.glob( os.path.join( exp_path,
                    "nnAnim-L{}*.npy".format(layer_no) ) ))

                if len(anim_files) > 0:
                    anim = ia.AnnotatedImage()
                    anim.load( file_name=anim_files[-1], file_path='' )
                    print("Loading anim: {}".format(anim_files[-1]))
                    print(" >> " + anim.__str__())
                    print("Creating annotations:")
                    anim.generate_cnn_annotations_cb(
                        min_size=min_size, max_size=max_size,
                        dilation_factor_centroids=dilation_factor_centroids,
                        dilation_factor_bodies=dilation_factor_bodies,
                        re_dilate_bodies=re_dilate_bodies )
                    ROIbase = os.path.join( exp_path, "nnROI{}".format(layer_no) )
                    anim.export_annotations_to_mat( file_name=ROIbase, file_path='')

            else:
                # Check if image and border exclusion files are present
                im_files = glob.glob( os.path.join( exp_path,
                    "*L{}-channels.mat".format(layer_no) ) )
                bdr_files = glob.glob( os.path.join( exp_path,
                    "*Borders{}.mat".format(layer_no) ) )

                if len(im_files) == 1 and len(bdr_files) == 1:

                    print("\n-------- Commencing nn-annotation --------")
                    print(exp_path)

                    # Load annotated image
                    print(" - Layer no: {}".format(layer_no))
                    anim = load_anim( im_files[0], bdr_files[0] )

                    # Segment annotated image
                    nn_annotate_anim( anim, body_net, centroid_net )

                    # Convert segmented image to annotations
                    print("Creating annotations:")
                    anim.generate_cnn_annotations_cb(
                        min_size=min_size, max_size=max_size,
                        dilation_factor_centroids=dilation_factor_centroids,
                        dilation_factor_bodies=dilation_factor_bodies,
                        re_dilate_bodies=re_dilate_bodies )

                    # Save classified image
                    save_annotations(anim, exp_path, layer_no)
