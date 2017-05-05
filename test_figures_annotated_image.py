#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:46:12 2017

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
filepath = '/data/DataSet_small1'
roifile = 'F03-Loc5-V1-20160202-ovlSplitROI1.mat'
imfile = 'F03-Loc5-V1-20160202-L1-channels.mat'

im_size = (27,27)
rotation_list = np.arange(0,360,1)
scale_list_x = np.arange(0.9,1.1,0.01)
scale_list_y = np.arange(0.9,1.1,0.01)
noise_level_list = np.arange(0.1,0.3,0.01)


# Create an instance of the AnnotatedImage class
anim1 = ia.AnnotatedImage()
anim1.import_annotations_from_mat(file_name=roifile,file_path=filepath)
anim1.add_image_from_file(file_name=imfile,file_path=filepath)
anim1.exclude_border = (40,40,40,40)
print(" >> " + anim1.__str__())

# Make RGB grid with 20 anim1 annotations
ch_ord = (0,1,2)
annot_no = []
annot_no.extend(range(239))
annot_no.extend(range(239))
annot_no.extend(range(239))
image_grid, shift = anim1.image_grid_RGB( im_size,
    annotation_nrs=annot_no, n_x=32, n_y=16, channel_order=ch_ord,
    line_color=0, auto_scale=True )
image_grid[:,:,2] = 0
bodies_grid, b_shift = anim1.image_grid_RGB( im_size, image_type='Bodies',
    annotation_nrs=list(range(16)), n_x=4, n_y=4, channel_order=ch_ord,
    line_color=1, auto_scale=True )


# Get training set
anim1.include_annotation_typenrs = 1
anim1.centroid_dilation_factor = 2
samples_mrph,labels_mrph,annotations_mrph = anim1.get_batch( \
        # im_size, annotation_type='bodies', return_annotations="bodies",
        im_size, annotation_type='centroids', return_annotations="centroids",
        m_samples=50,  morph_annotations=False,
        annotation_border_ratio=0.2,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list )

an_lst = []
for x in range(0,25,5):
    an_lst.extend(list(range(x+0,x+5)))
    an_lst.extend(list(range(x+25,x+30)))
print(an_lst)
samples_grid,_,brdr = ia.image_grid_RGB( samples_mrph,
    n_channels=anim1.n_channels, annotation_nrs=an_lst,
    image_size=im_size, n_x=10, n_y=5, channel_order=ch_ord,
    amplitude_scaling=(1.33,1.33,1.0), line_color=0,
    auto_scale=True, return_borders=True )
samples_grid[:,:,2] = 0
samples_grid[brdr==1] = 1 # make borders white
annotations_grid,shift,_ = ia.image_grid_RGB( annotations_mrph,
    n_channels=1, annotation_nrs=an_lst,
    image_size=im_size, n_x=10, n_y=5, channel_order=ch_ord,
    amplitude_scaling=(1.33,1.33,1), line_color=1,
    auto_scale=True, return_borders=True )
brdr[:,int(brdr.shape[1]/2):,:] = 0
annotations_grid[brdr==1] = 0.5 # make borders white

grid = np.concatenate([samples_grid,annotations_grid],axis=0)

anim1.crop(left=200, top=150, width=100, height=100 )

# ************************************************************
# Show matplotlib images

# Show channels
# with sns.axes_style("white"):
#     fig,ax = plt.subplots(figsize=(12,6), facecolor='w', edgecolor='w')
#     ax.imshow( image_grid, interpolation='nearest', vmax=image_grid.max()*0.7 )
#     plt.axis('tight')
#     plt.axis('off')
#
# with sns.axes_style("white"):
#     fig,ax = plt.subplots(figsize=(8,8), facecolor='w', edgecolor='w')
#     ax.imshow( anim1.RGB(channel_order=(0,1,2),amplitude_scaling=(1.5,1.5,0)), interpolation='nearest')
#     plt.axis('tight')
#     plt.axis('off')
#
# with sns.axes_style("white"):
#     fig,ax = plt.subplots(figsize=(8,8), facecolor='w', edgecolor='w')
#     ax.imshow( anim1.RGB(channel_order=(0,1,2),amplitude_scaling=(1.5,1.5,0)), interpolation='nearest')
#     for an in anim1.annotation:
#         ax.plot( an.perimeter[:,1], an.perimeter[:,0],
#             linewidth=1, color="#ffffff" )
#     plt.axis('tight')
#     plt.axis('off')
#
# with sns.axes_style("white"):
#     fig,ax = plt.subplots(figsize=(8,8), facecolor='w', edgecolor='w')
#     ax.imshow( anim1.bodies>0, interpolation='nearest')
#     plt.axis('tight')
#     plt.axis('off')
#
# with sns.axes_style("white"):
#     fig,ax = plt.subplots(figsize=(8,8), facecolor='w', edgecolor='w')
#     ax.imshow( anim1.centroids>0, interpolation='nearest')
#     plt.axis('tight')
#     plt.axis('off')

with sns.axes_style("white"):
    fig,ax = plt.subplots(figsize=(8,8), facecolor='w', edgecolor='w')
    ax.imshow( grid, interpolation='nearest')
    plt.axis('tight')
    plt.axis('off')

# Show plots
plt.tight_layout()
plt.show()
