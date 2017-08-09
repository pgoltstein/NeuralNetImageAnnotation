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

# Settings
network_path = 'D:/neuralnets'
body_network_name = "bnet002"
centroid_network_name = "cnet002"
use_channels = [0,1,3]
min_size = 100
max_size = 1000
dilation_factor_centroids = -1
dilation_factor_bodies = 0
re_dilate_bodies = 0
normalize_images = True


########################################################################
### Load image & border data
im_file = glob.glob(file_stem+"*channels.mat")[layer_no]
bdr_file = glob.glob(file_stem+"*Border*.mat")[layer_no]
print("\n-------- Commencing nn-annotation --------")
print("Layer no: {}".format(layer_no))
anim = ia.AnnotatedImage()
print("Importing image: {}".format(im_file))
anim.add_image_from_file(file_name=im_file,file_path='',
                normalize=normalize_images, use_channels=use_channels)
print("Setting border excusion: {}".format(bdr_file))
anim.exclude_border = bdr_file
print(" >> " + anim.__str__())
