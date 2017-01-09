#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:46:12 2017

@author: pgoltstein
"""

import ImageAnnotation as ia
import numpy as np
import importlib
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat,savemat


importlib.reload(ia)

anim = ia.AnnotatedImage()
print(anim)

filepath = '/Users/pgoltstein/Dropbox/TEMP/DataSet2'
roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'

anim.import_annotations_from_mat(file_name=roifile,file_path=filepath)
#anim.import_annotations_from_mat(file_name='0test_rois.mat',file_path=filepath)
print(anim)

anim.add_image_from_file(file_name=imfile,file_path=filepath)
print(anim)
#
#plt.figure()
#plt.imshow(anim.channel[0],interpolation='nearest')
#
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

samples,labels,annotations = anim.annotation_detection_training_batch( 
                dilation_factor=-3, m_samples=500, morph_annotations=True )

lin_im_mat = samples[labels[:,1]==0,:]
grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5 )
f, ax = plt.subplots(figsize=(12,6))
ax.imshow(grid, interpolation='nearest')
ax.set_title("negatives")

lin_im_mat = annotations[labels[:,1]==0,:]
grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5, line_color=1 )
f, ax = plt.subplots(figsize=(12,6))
ax.imshow(grid, interpolation='nearest')
ax.set_title("negatives")

lin_im_mat = samples[labels[:,1]==1,:]
grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5 )
f, ax = plt.subplots(figsize=(12,6))
ax.imshow(grid, interpolation='nearest')
ax.set_title("positives")

lin_im_mat = annotations[labels[:,1]==1,:]
grid=anim.image_grid_RGB( lin_im_mat[0:49,:], n_x=10, n_y=5, line_color=1 )
f, ax = plt.subplots(figsize=(12,6))
ax.imshow(grid, interpolation='nearest')
ax.set_title("positives")

    
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



