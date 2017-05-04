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

# ************************************************************
# Show matplotlib images

# Show channels
with sns.axes_style("white"):
    fig,ax = plt.subplots(figsize=(12,6), facecolor='w', edgecolor='w')
    ax.imshow( image_grid, interpolation='nearest', vmax=image_grid.max()*0.7 )
    plt.axis('tight')
    plt.axis('off')

with sns.axes_style("white"):
    fig,ax = plt.subplots(figsize=(8,8), facecolor='w', edgecolor='w')
    ax.imshow( anim1.RGB(channel_order=(0,1,2),amplitude_scaling=(1.5,1.5,0)), interpolation='nearest')
    plt.axis('tight')
    plt.axis('off')

# Show plots
plt.tight_layout()
plt.show()
