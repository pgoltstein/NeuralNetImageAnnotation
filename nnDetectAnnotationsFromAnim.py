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
min_size = 150
max_size = 1000
dilation_factor_centroids = -1
dilation_factor_bodies = 0
re_dilate_bodies = 0
normalize_images = True

# Find nnAnIm file
anim_files = glob.glob( os.path.join( datapath,
    "nnAnim-L{}*.npy".format(layer_no) ) )
if len(anim_files) > 0:

    # Detect annotations
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
