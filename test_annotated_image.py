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
filepath = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
im_size = (31,31)
rotation_list = np.arange(0,360,1)
scale_list_x = np.arange(0.9,1.1,0.01)
scale_list_y = np.arange(0.9,1.1,0.01)
noise_level_list = np.arange(0,0.05,0.001)

# Create an instance of the AnnotatedImage class
print(" ")
print("Create an instance of the AnnotatedImage class named: anim1")
anim1 = ia.AnnotatedImage()
print("String output of anim1:")
print(" >> " + anim1.__str__())

# Import annotations from ROI file (matlab)
roifile = 'F03-Loc5-V1-20160209-ovlSplitROI1.mat'
print(" ")
print("Importing annotations from ROI file (matlab) to anim1:")
print(os.path.join(filepath,roifile))
anim1.import_annotations_from_mat(file_name=roifile,file_path=filepath)
print(" >> " + anim1.__str__())

# Add an image to anim
imfile = 'F03-Loc5-V1-20160209-OverlayL1.tiff'
print(" ")
print("Importing image from tiff file to anim1:")
print(os.path.join(filepath,roifile))
anim1.add_image_from_file(file_name=imfile,file_path=filepath)
print(" >> " + anim1.__str__())

# Save anim
print(" ")
print("Saving AnnotatedImage to testAnim.npy")
anim1.save('testAnim.npy',filepath)

# Load anim
print(" ")
print("Define new annotated image (anim2), with downsample=2 and load data from testAnim.anim")
anim2 = ia.AnnotatedImage(downsample=2)
anim2.load('testAnim.npy',filepath)
print("String output of anim2:")
print(" >> " + anim2.__str__())

# # Dilate bodies
# print(" ")
# print("Changing dilation factor of anim2 bodies to -3")
# anim2.body_dilation_factor = -3

# Dilate centroids
print(" ")
print("Changing dilation factor of anim2 centroids to 2")
anim2.centroid_dilation_factor = 2

print("Class labels that are set for training: {}".format(anim2.class_labels))
print("anim2.include_annotation_typenrs: {}".format(anim2.include_annotation_typenrs))

# Make RGB grid with 20 anim1 annotations
print(" ")
print("Get RGB image grid with 16 annotations from anim2")
image_grid, shift = anim2.image_grid_RGB( im_size,
    annotation_nrs=list(range(16)), n_x=4, n_y=4,
    line_color=0, auto_scale=True )
image_grid[:,:,2] = 0
bodies_grid, b_shift = anim2.image_grid_RGB( im_size, image_type='Bodies',
    annotation_nrs=list(range(16)), n_x=4, n_y=4,
    line_color=1, auto_scale=True )

# Get training set
print(" ")
print("Get training set with non-morphed annotations from anim2")
annot_show_list = list(range(16))
samples,labels,annotations = anim2.get_batch( \
        im_size, annotation_type='bodies', return_annotations='bodies',
        m_samples=len(annot_show_list), morph_annotations=False,
        annotation_border_ratio=0.5 )

samples_grid,_ = ia.image_grid_RGB( samples,
    n_channels=anim2.n_channels, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=4, n_y=4, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0, auto_scale=True )
samples_grid[:,:,2] = 0
annotations_grid,shift_grid = ia.image_grid_RGB( annotations,
    n_channels=1, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=4, n_y=4, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=1, auto_scale=True )

# Get training set
print(" ")
print("Get training set with morphed annotations from anim2")
samples_mrph,labels_mrph,annotations_mrph = anim2.get_batch( \
        im_size, annotation_type='bodies', return_annotations="bodies",
        m_samples=len(annot_show_list),  morph_annotations=True,
        annotation_border_ratio=0.5,
        rotation_list=rotation_list, scale_list_x=scale_list_x,
        scale_list_y=scale_list_y, noise_level_list=noise_level_list )

samples_grid_mrph,_ = ia.image_grid_RGB( samples_mrph,
    n_channels=anim2.n_channels, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=4, n_y=4, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0, auto_scale=True )
samples_grid_mrph[:,:,2] = 0
annotations_grid_mrph,shift_mrph = ia.image_grid_RGB( annotations_mrph,
    n_channels=1, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=4, n_y=4, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=1, auto_scale=True )

# Save anim
print(" ")
print("Exporting anim1 annotations to zzROIpy.mat")
anim1.export_annotations_to_mat( 'zzROIpy.mat', file_path=filepath)

# Load anim
print(" ")
print("Define new annotated image (anim3) and copy image data from anim1")
anim3 = ia.AnnotatedImage(image_data=anim1.channel)
print("String output of anim3:")
print(" >> " + anim3.__str__())

# Import annotations from zzROI file (matlab)
print(" ")
print("Importing annotations from zzROIpy.mat file to anim3:")
print(os.path.join(filepath,'zzROIpy.mat'))
anim3.import_annotations_from_mat(file_name='zzROIpy.mat',file_path=filepath)
print(" >> " + anim3.__str__())


# Save anim2 annotations
print(" ")
print("Exporting anim2 annotations to zzROIpy2.mat")
anim2.export_annotations_to_mat( 'zzROIpy2.mat', file_path=filepath)

# Load anim
print(" ")
print("Define new annotated image (anim4) and copy image data from anim1")
anim4 = ia.AnnotatedImage(image_data=anim1.channel)
print("String output of anim4:")
print(" >> " + anim4.__str__())

# Import annotations from zzROI2 file (matlab)
print(" ")
print("Importing annotations from zzROIpy2.mat file to anim4:")
print(os.path.join(filepath,'zzROIpy2.mat'))
anim4.import_annotations_from_mat(file_name='zzROIpy2.mat',file_path=filepath)
print(" >> " + anim4.__str__())


# ************************************************************
# Show matplotlib images

# Show channels
with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax = list(range(3))
    for ch in range(3):
        ax[ch] = plt.subplot2grid( (2,3), (0,ch) )
        ax[ch].imshow( anim1.channel[ch], cmap='gray',
            interpolation='nearest', vmax=anim1.channel[ch].max()*0.7 )
        ax[ch].set_title("anim: Channel{}".format(ch))
        plt.axis('tight')
        plt.axis('off')

    axb = plt.subplot2grid( (2,3), (1,0) )
    axb.imshow(anim1.bodies,vmin=-0.1)
    axb.set_title("anim: Annotated bodies")
    plt.axis('tight')
    plt.axis('off')

    axc = plt.subplot2grid( (2,3), (1,1) )
    axc.imshow(anim1.centroids,vmin=-0.1)
    axc.set_title("anim: Annotated bodies")
    plt.axis('tight')
    plt.axis('off')

    axr = plt.subplot2grid( (2,3), (1,2) )
    axr.imshow(anim1.RGB(),interpolation='nearest')
    for an in anim1.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim: RGB with annotations")
    plt.axis('tight')
    plt.axis('off')

# Show channels
with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax = list(range(3))
    for ch in range(3):
        ax[ch] = plt.subplot2grid( (2,3), (0,ch) )
        ax[ch].imshow( anim2.channel[ch], cmap='gray',
            interpolation='nearest', vmax=anim2.channel[ch].max()*0.7 )
        ax[ch].set_title("anim2: Channel{}".format(ch))
        plt.axis('tight')
        plt.axis('off')

    axb = plt.subplot2grid( (2,3), (1,0) )
    axb.imshow(anim2.bodies,vmin=-0.1)
    axb.set_title("anim2: Annotated bodies")
    plt.axis('tight')
    plt.axis('off')

    axc = plt.subplot2grid( (2,3), (1,1) )
    axc.imshow(anim2.centroids,vmin=-0.1)
    axc.set_title("anim2: Annotated bodies")
    plt.axis('tight')
    plt.axis('off')

    axr = plt.subplot2grid( (2,3), (1,2) )
    axr.imshow(anim2.RGB(),interpolation='nearest')
    for an in anim2.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim2: RGB with annotations")
    plt.axis('tight')
    plt.axis('off')

# Show channels
with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax1 = plt.subplot2grid( (2,3), (0,0) )
    ax1.imshow( image_grid, interpolation='nearest', vmax=image_grid.max()*0.8 )
    for nr in range(16):
        an = anim2.annotation[nr]
        ax1.plot( shift[nr][1]+(an.perimeter[:,1]-an.x),
                  shift[nr][0]+(an.perimeter[:,0]-an.y),
                    linewidth=1, color="#ffffff" )
    ax1.set_title("First 16 annotations in anim2 (image)")
    plt.axis('tight')
    plt.axis('off')

    ax2 = plt.subplot2grid( (2,3), (1,0) )
    ax2.imshow( bodies_grid, interpolation='nearest', vmax=bodies_grid.max() )
    for nr in range(16):
        an = anim2.annotation[nr]
        ax2.plot( shift[nr][1]+(an.perimeter[:,1]-an.x),
                  shift[nr][0]+(an.perimeter[:,0]-an.y),
                    linewidth=1, color="#ff0000" )
    ax2.set_title("First 16 annotations in anim2 (body)")
    plt.axis('tight')
    plt.axis('off')

    ax3 = plt.subplot2grid( (2,3), (0,1) )
    ax3.imshow( samples_grid, interpolation='nearest', vmax=image_grid.max()*0.8 )
    ax3.set_title("First 16 training samples (image)")
    plt.axis('tight')
    plt.axis('off')

    ax4 = plt.subplot2grid( (2,3), (1,1) )
    ax4.imshow( annotations_grid, interpolation='nearest', vmax=image_grid.max()*0.8 )
    ax4.set_title("First 16 training samples (body)")
    for cnt,nr in enumerate(annot_show_list):
        if labels[cnt,0] != 1:
            an = ia.Annotation( ia.vec2image( annotations[nr,:],
                                        n_channels=1, image_size=im_size ) )
            ax4.plot( 0.5+shift_grid[cnt][1]+(an.perimeter[:,1]-(im_size[1]/2)),
                      0.5+shift_grid[cnt][0]+(an.perimeter[:,0]-(im_size[1]/2)),
                        linewidth=1, color="#ff0000" )
    plt.axis('tight')
    plt.axis('off')

    ax5 = plt.subplot2grid( (2,3), (0,2) )
    ax5.imshow( samples_grid_mrph, interpolation='nearest', vmax=image_grid.max()*0.8 )
    ax5.set_title("First 16 morphed training samples (image)")
    plt.axis('tight')
    plt.axis('off')

    ax6 = plt.subplot2grid( (2,3), (1,2) )
    ax6.imshow( annotations_grid_mrph, interpolation='nearest', vmax=image_grid.max()*0.8 )
    for cnt,nr in enumerate(annot_show_list):
        if labels_mrph[cnt,0] != 1:
            an = ia.Annotation( ia.vec2image( annotations_mrph[nr,:],
                                        n_channels=1, image_size=im_size ) )
            ax6.plot( 0.5+shift_mrph[cnt][1]+(an.perimeter[:,1]-(im_size[1]/2)),
                      0.5+shift_mrph[cnt][0]+(an.perimeter[:,0]-(im_size[1]/2)),
                        linewidth=1, color="#ff0000" )
    ax6.set_title("First 16 morphed training samples (body)")
    plt.axis('tight')
    plt.axis('off')

# Show channels
with sns.axes_style("white"):
    plt.figure(figsize=(12,6), facecolor='w', edgecolor='w')
    axr = plt.subplot2grid( (1,2), (0,0) )
    axr.imshow(anim1.RGB(),interpolation='nearest')
    for an in anim3.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim1 with annotations of anim3")
    plt.axis('tight')
    plt.axis('off')

    axr = plt.subplot2grid( (1,2), (0,1) )
    axr.imshow(anim3.RGB(),interpolation='nearest')
    for an in anim1.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim3 with annotations of anim1")
    plt.axis('tight')
    plt.axis('off')

# Show channels anim1 anim4
with sns.axes_style("white"):
    plt.figure(figsize=(12,6), facecolor='w', edgecolor='w')
    axr = plt.subplot2grid( (1,2), (0,0) )
    axr.imshow(anim1.RGB(),interpolation='nearest')
    for an in anim4.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim1 with annotations of anim4")
    plt.axis('tight')
    plt.axis('off')

    axr = plt.subplot2grid( (1,2), (0,1) )
    axr.imshow(anim4.RGB(),interpolation='nearest')
    for an in anim1.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim4 with annotations of anim1")
    plt.axis('tight')
    plt.axis('off')

# Show plots
plt.tight_layout()
plt.show()
