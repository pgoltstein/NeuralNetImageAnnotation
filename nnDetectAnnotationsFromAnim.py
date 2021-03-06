#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 09 2017

Detects nnROI annotations from annotated image

@author: pgoltstein
"""

### Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ImageAnnotation as ia
import argparse
import os
import glob
from scipy import ndimage
from skimage import measure

# Arguments
parser = argparse.ArgumentParser( \
    description = \
        "Detects annotations from neural network annotated AnIm files " + \
        "located in the supplied directory " + \
        "(written by Pieter Goltstein - August 2017)")

parser.add_argument('datapath', type=str,
    help= 'Path to folder that contains AnnotatedImage class files')
parser.add_argument('-l', '--multilevel_layer_no', type=int, default=1,
                    help= 'Number of the multilevel layer to load (default=1)')
args = parser.parse_args()

# Settings
datapath = str(args.datapath)
layer_no = args.multilevel_layer_no
min_size = 120
max_size = 750
dilation_factor_centroids = 0
dilation_factor_bodies = 0
re_dilate_bodies = 0
im_norm_perc = 99

# ************************************************************
# Find nnAnIm file and detect annotations
anim_files = sorted(glob.glob( os.path.join( datapath,
    "nnAnIm-L{}*.npy".format(layer_no) ) ))
if len(anim_files) == 0:
    print("No image found that matches 'nnAnIm-L{}*.npy' ...".format(layer_no))
else:
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
    ROIbase = os.path.join( datapath, "nnROI{}".format(layer_no) )
    anim.export_annotations_to_mat( file_name=ROIbase, file_path='')

    # ************************************************************
    # Show matplotlib images
    anRGB = np.zeros((anim.detected_bodies.shape[0],anim.detected_bodies.shape[1],3))
    anRGB[:,:,1] = anim.detected_centroids
    anRGB[:,:,2] = anim.detected_bodies

    imRGB = anim.RGB().astype(np.float)
    for ch in range(3):
        norm_perc = np.percentile( imRGB[:,:,ch], im_norm_perc )
        imRGB[:,:,ch] = imRGB[:,:,ch] / norm_perc
    imRGB[imRGB>1] = 1
    imRGB[imRGB<0] = 0

    # Show image and classification result
    with sns.axes_style("white"):
        plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
        axr = plt.subplot2grid( (1,2), (0,0) )
        axr.imshow( imRGB, interpolation='nearest')
        axr.set_title("Image")
        plt.axis('tight')
        plt.axis('off')

        axb = plt.subplot2grid( (1,2), (0,1) )
        axb.imshow(anRGB)
        axb.set_title("Annotated bodies and centroids")
        plt.axis('tight')
        plt.axis('off')

    with sns.axes_style("white"):
        plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
        axr = plt.subplot2grid( (1,2), (0,0) )
        axr.imshow( imRGB, interpolation='nearest')
        for an in anim.annotation:
            axr.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ffffff" )
        axr.set_title("Annotated image")
        plt.axis('tight')
        plt.axis('off')

        axr = plt.subplot2grid( (1,2), (0,1) )
        axr.imshow(anRGB)
        for an in anim.annotation:
            axr.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ffffff" )
        axr.set_title("Annotated image")
        plt.axis('tight')
        plt.axis('off')

    print('Done!\n')
    plt.show()
