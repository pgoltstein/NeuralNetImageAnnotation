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
from scipy.io import loadmat,savemat
print(" ")
print("----------------------------------------------------------")
print("Importing ImageAnnotation as ia")
import ImageAnnotation as ia

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Overall settings
filepath = '/Users/pgoltstein/Dropbox/TEMP/DataSet2'
im_size = (31,31)
rotation_list = np.array(range(360))
scale_list_x = np.array(range(500,1500)) / 1000
scale_list_y = np.array(range(500,1500)) / 1000
noise_level_list = np.array(range(25)) / 1000

# Create an instance of the AnnotatedImage class
print(" ")
print("Create an instance of the AnnotatedImage class named: anim1")
anim1 = ia.AnnotatedImage()
print("String output of anim1:")
print(" >> " + anim1.__str__())

# Import annotations from ROI file (matlab)
roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
print(" ")
print("Importing annotations from ROI file (matlab) to anim1:")
print(os.path.join(filepath,roifile))
anim1.import_annotations_from_mat(file_name=roifile,file_path=filepath)
print(" >> " + anim1.__str__())

# Add an image to anim
imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'
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
print("Define new annotated image (anim2) and load data from testAnim.anim")
anim2 = ia.AnnotatedImage()
anim2.load('testAnim.npy',filepath)
print("String output of anim2:")
print(" >> " + anim1.__str__())

# # Dilate bodies
# print(" ")
# print("Changing dilation factor of anim2 bodies to -3")
# anim2.body_dilation_factor = -3

# Dilate centroids
print(" ")
print("Changing dilation factor of anim2 centroids to 2")
anim2.centroid_dilation_factor = 2

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
        im_size, annotation_type='centroids', return_annotations=True,
        m_samples=100, exclude_border=(0,0,0,0), morph_annotations=False )
samples_grid,_ = ia.image_grid_RGB( samples,
    n_channels=anim2.n_channels, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=4, n_y=4, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=0, auto_scale=True )
samples_grid[:,:,2] = 0
annotations_grid,_ = ia.image_grid_RGB( annotations,
    n_channels=1, annotation_nrs=annot_show_list,
    image_size=im_size, n_x=4, n_y=4, channel_order=(0,1,2),
    amplitude_scaling=(1.33,1.33,1), line_color=1, auto_scale=True )

# Get training set
print(" ")
print("Get training set with morphed annotations from anim2")
samples_mrph,labels_mrph,annotations_mrph = anim2.get_batch( \
        im_size, annotation_type='centroids', return_annotations=True,
        m_samples=100, exclude_border=(0,0,0,0),  morph_annotations=True,
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
        an = ia.Annotation( ia.vec2image( annotations_mrph[nr,:],
                                    n_channels=1, image_size=im_size ) )
        ax6.plot( shift_mrph[cnt][1]+(an.perimeter[:,1]-(im_size[1]/2)),
                  shift_mrph[cnt][0]+(an.perimeter[:,0]-(im_size[1]/2)),
                    linewidth=1, color="#ff0000" )
    ax6.set_title("First 16 morphed training samples (body)")
    plt.axis('tight')
    plt.axis('off')


#plt.figure()
#plt.imshow(anim.bodies(),interpolation='nearest')
#
#ims = anim.zoom(114,213)
#_, ax = plt.subplots(1,len(ims))
#for nr,im in enumerate(ims):
#    ax[nr].imshow(im,interpolation='nearest')
#
#lin_im = anim.zoom_1d(114,213)
#
#ims = anim.image_list_1d_to_2d( lin_im )
#_, ax = plt.subplots(1,len(ims))
#for nr,im in enumerate(ims):
#    ax[nr].imshow(im,interpolation='nearest')
#
# samples,labels,annotations = anim.annotation_detection_training_batch(
#                 dilation_factor=-3, m_samples=500, morph_annotations=True )
#
# lin_im_mat = samples[labels[:,1]==0,:]
# grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5 )
# f, ax = plt.subplots(figsize=(12,6))
# ax.imshow(grid, interpolation='nearest')
# ax.set_title("negatives")
#
# lin_im_mat = annotations[labels[:,1]==0,:]
# grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5, line_color=1 )
# f, ax = plt.subplots(figsize=(12,6))
# ax.imshow(grid, interpolation='nearest')
# ax.set_title("negatives")
#
# lin_im_mat = samples[labels[:,1]==1,:]
# grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5 )
# f, ax = plt.subplots(figsize=(12,6))
# ax.imshow(grid, interpolation='nearest')
# ax.set_title("positives")
#
# lin_im_mat = annotations[labels[:,1]==1,:]
# grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5, line_color=1 )
# f, ax = plt.subplots(figsize=(12,6))
# ax.imshow(grid, interpolation='nearest')
# ax.set_title("positives")


#image = anim.channel[0]
#image[anim.centroids(dilation_factor=5)==1]=0
#plt.figure()
#plt.imshow(image,interpolation='nearest')


#anim.export_annotations_to_mat(file_name='0test_rois',file_path=filepath)

#mat_data = loadmat(os.path.join(filepath,'0test_rois'))
#mat_data2 = loadmat(os.path.join(filepath,roifile))


#anim2 = ia.AnnotatedImage()
#print(anim2)
#
#filepath = '/Users/pgoltstein/Dropbox/TEMP'
#roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
#imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'
#
#anim2.import_annotations_from_mat(file_name=roifile,file_path=filepath)


# Show plots
# sns.set_context("talk")
plt.tight_layout()
plt.show()
