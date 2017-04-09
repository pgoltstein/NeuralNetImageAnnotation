#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:38:41 2017

@author: pgoltstein
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(" ")
print("----------------------------------------------------------")
print("Importing ImageAnnotation as ia")
import ImageAnnotation as ia

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Overall settings
data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet_small'
im_size = (31,31)
rotation_list = np.array(range(360))
scale_list_x = np.array(range(500,1500)) / 1000
scale_list_y = np.array(range(500,1500)) / 1000
noise_level_list = np.array(range(25)) / 1000

# Create an instance of the AnnotatedImage class
print(" ")
print("Create an instance of the AnnotatedImageSet class named: ais1")
ais1 = ia.AnnotatedImageSet()
print("String output of ais1:")
print(" >> " + ais1.__str__())

print(" ")
print("Loading data from directory into ais1:")
print(data_path)
ais1.load_data_dir_tiff_mat(data_path)
print(" >> " + ais1.__str__())

# Dilate centroids
print(" ")
print("Changing dilation factor of ais1 centroids to 0")
ais1.centroid_dilation_factor = 0

# Get training set
print(" ")
print("Get training set with non-morphed annotations from ais1")
samples,labels,annotations = ais1.data_sample( \
        im_size, annotation_type='bodies', return_annotations=True,
        m_samples=30, exclude_border=(0,0,0,0), morph_annotations=False )

print(" ")
print("Construct RGB grid from first 16 annotations in training set")
annot_show_list = list(range(30))
samples_grid,_ = ia.image_grid_RGB( samples,
    n_channels=ais1.n_channels, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=10, n_y=3, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0, auto_scale=True )
samples_grid[:,:,2] = 0
annotations_grid,_ = ia.image_grid_RGB( annotations,
    n_channels=1, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=10, n_y=3, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0.5, auto_scale=True )

# ************************************************************
# Show matplotlib images

# Show channels
with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax1 = plt.subplot2grid( (2,1), (0,0) )
    ax1.imshow( samples_grid, interpolation='nearest', vmax=samples_grid.max()*0.8 )
    ax1.set_title("First 16 annotations in ais1 (image)")
    plt.axis('tight')
    plt.axis('off')

    ax2 = plt.subplot2grid( (2,1), (1,0) )
    ax2.imshow( annotations_grid, interpolation='nearest', vmax=annotations_grid.max() )
    ax2.set_title("First 16 annotations in ais1 (centroid)")
    plt.axis('tight')
    plt.axis('off')

# Show plots
plt.tight_layout()
plt.show()
