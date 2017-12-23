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
import time, datetime
import os

print(" ")
print("----------------------------------------------------------")
print("Importing ImageAnnotation as ia")
import ImageAnnotation as ia

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Overall settings
# data_path = '/data/EGR/Dataset1'
data_path = '/data/roi/DataSet_small'
im_size = (41,41)
annot_size = (41,41)
rotation_list = np.arange(0,360,1)
scale_list_x = np.arange(0.9,1.1,0.01)
scale_list_y = np.arange(0.9,1.1,0.01)
noise_level_list = np.arange(0.1,0.3,0.01)

# Create an instance of the AnnotatedImage class
print(" ")
print("Create an instance of the AnnotatedImageSet class named: ais1")
ais1 = ia.AnnotatedImageSet(downsample=None)
print("String output of ais1:")
print(" >> " + ais1.__str__())

print(" ")
print("Loading data from directory into ais1:")
print(data_path)
# ais1.load_data_dir_tiff_mat(data_path, exclude_border="load")
ais1.load_data_dir_tiff_mat(data_path, exclude_border=(10,20,30,40))
print(" >> " + ais1.__str__())

# Dilate centroids
print(" ")
print("Changing dilation factor of ais1 centroids to 0")
ais1.centroid_dilation_factor = 0

print("ais1.include_annotation_typenrs = [1,4]")
ais1.include_annotation_typenrs = [1,4]
print("Class labels that are set for training: {}".format(ais1.class_labels))

print("ais1.include_annotation_typenrs = 1")
ais1.include_annotation_typenrs = 1
print("Class labels that are set for training: {}".format(ais1.class_labels))

print("ais1.include_annotation_typenrs = None")
ais1.include_annotation_typenrs = None
print("Class labels that are set for training: {}".format(ais1.class_labels))

# Dilate centroids
print(" ")
print("Included type_nrs: {}".format(ais1.include_annotation_typenrs))
# print("Changing annotation type nr of ais1 centroids to [1,2,3,4]")
# ais1.include_annotation_typenrs = [1,2,3,4]
# print("Included type_nrs: {}".format(ais1.include_annotation_typenrs))
# print("Changing annotation type nr of ais1 centroids to 1")
# ais1.include_annotation_typenrs = 1
# print("Included type_nrs: {}".format(ais1.include_annotation_typenrs))
# print("Changing annotation type nr of ais1 centroids to [1,4]")
# ais1.include_annotation_typenrs = [1,4]
# print("Included type_nrs: {}".format(ais1.include_annotation_typenrs))

# Crop image
print("Cropping image")
ais1.crop(100,100,300,300)

# Get non-morphed training set
m_samples = 1000
print(" ")
print("Get training set (m={}) with non-morphed annotations from ais1".format(m_samples))
t_start = time.time()
samples,labels,annotations = ais1.data_sample( \
        im_size, annotation_type='bodies', return_annotations=False,
        m_samples=m_samples, morph_annotations=False,
        sample_ratio=(0.4,0.4,0.2),
        annotation_border_ratio=0.43 )
t_curr = time.time()
print(' -- Duration = {:.0f} ms'.format(1000*(t_curr-t_start)) )

# Get training set
print(" ")
print("Get anothertraining set (m={}) with non-morphed annotations from ais1".format(m_samples))
t_start = time.time()
samples,labels,annotations = ais1.data_sample( \
        im_size, annotation_type='centroids', return_annotations='centroids',
        m_samples=m_samples, morph_annotations=False,
        sample_ratio=(0.4,0.4,0.2),
        annotation_border_ratio=0.43 )
t_curr = time.time()
print(' -- Duration = {:.0f} ms'.format(1000*(t_curr-t_start)) )

# Get training set
print(" ")
print("Get small training set with morphed annotations from ais1")
samples,labels,annotations = ais1.data_sample( \
        im_size, annotation_type='bodies',
        return_size=annot_size, return_annotations='bodies',
        m_samples=30,
        # sample_ratio=None,
        sample_ratio=(0.5,0.5,0),
        annotation_border_ratio=0.5, normalize_samples=True,
        segment_all=False, morph_annotations=False,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list )
# print(labels)

print(" ")
print("Construct RGB grid from first 16 annotations in training set")
annot_show_list = list(range(30))
samples_grid,_,brdrs = ia.image_grid_RGB( samples,
    n_channels=ais1.n_channels, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=10, n_y=3, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0,
    return_borders=True, auto_scale=True )
samples_grid[:,:,2] = 0
samples_grid[brdrs==1] = 1
annotations_grid,_,brdrs = ia.image_grid_RGB( annotations,
    n_channels=1, annotation_nrs=annot_show_list,
    image_size=annot_size, n_x=10, n_y=3, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0.5,
    return_borders=True, auto_scale=True )
annotations_grid[brdrs==1] = 1

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
