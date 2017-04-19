#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2017

Compares annotations of an AnnotatedImage or AnnotatedImageSet with
the annotations of a 'ground thruth' AnnotatedImage or AnnotatedImageSet
and outputs statistics and visualization off the performance

@author: pgoltstein
"""

########################################################################
### Imports
########################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ImageAnnotation as ia
import argparse, os, glob
from scipy import ndimage
from skimage import measure


########################################################################
# Other variables
max_centroid_distance = 5
max_body_centroid_distance = 30
min_body_overlap_fraction = 0.66
min_size = 80
max_size = 1000
dilation_factor_centroids = -2
dilation_factor_bodies = 0
re_dilate_bodies = 0
# file_stem = "F03-Loc5-V1-20160202"
# file_stem = "F04-Loc5-V1-20160617"
file_stem = "/Users/pgoltstein/Dropbox/TEMP/DataSet2/F04-Loc5-V1-20160202"
# file_stem = "/Users/pgoltstein/Dropbox/TEMP/DataSet2/F02-Loc4-P-20160320"
# file_stem = "/Users/pgoltstein/Dropbox/TEMP/DataSet2/21a-Loc1-V1-20160618"
# file_stem = "/Users/pgoltstein/Dropbox/TEMP/DataSet2/21b-Loc3-RL-20160406"

########################################################################
### Load Ground Truth AnnotatedImage
im_file_gt = glob.glob(file_stem+"*channels.mat")[0]
# im_file_gt = glob.glob(file_stem+"*.tiff")[0]
bdr_file_gt = glob.glob(file_stem+"*Border*.mat")[0]
roi_file_gt = glob.glob(file_stem+"*ovlSplitROI*.mat")[0]
print("\n-------- Ground Truth AnnotatedImage --------")
anim_gt = ia.AnnotatedImage()
print("Importing image: {}".format(im_file_gt))
anim_gt.add_image_from_file(file_name=im_file_gt,file_path='')
print("Importing ROI's: {}".format(roi_file_gt))
anim_gt.import_annotations_from_mat(file_name=roi_file_gt,file_path='')
print("Setting border excusion: {}".format(bdr_file_gt))
anim_gt.exclude_border = bdr_file_gt
print(" >> " + anim_gt.__str__())


########################################################################
### Load Neural Net AnnotatedImage
anim_file_nn = glob.glob(file_stem+"*anim*.npy")[-1]
bdr_file_nn = glob.glob(file_stem+"*Border*.mat")[0]
print("\n-------- Neural Net AnnotatedImage --------")
anim_nn = ia.AnnotatedImage()
print("Loading: {}".format(anim_file_nn))
anim_nn.load( anim_file_nn )
print("Setting border excusion: {}".format(bdr_file_nn))
anim_nn.exclude_border = bdr_file_nn
print(" >> " + anim_nn.__str__())


########################################################################
### Re-create Neural Net annotations
print("\n-------- Re-create NN annotations --------")
anim_nn.generate_cnn_annotations_cb( min_size=min_size, max_size=max_size,
    dilation_factor_centroids=dilation_factor_centroids,
    dilation_factor_bodies=dilation_factor_bodies,
    re_dilate_bodies=re_dilate_bodies )
print(" >> " + anim_nn.__str__())


########################################################################
### Calculate pixel based performance on bodies
print("\n-------- Pixel based performance --------")

# Calculate true/false pos/neg
gt_bodies = anim_gt.bodies>0
nn_bodies = anim_nn.bodies>0
n_pixels = anim_nn.bodies.size
true_pos = np.sum( nn_bodies[gt_bodies==1]==1 )
false_pos = np.sum( nn_bodies[gt_bodies==0]==1 )
false_neg = np.sum( nn_bodies[gt_bodies==1]==0 )
true_neg = np.sum( nn_bodies[gt_bodies==0]==0 )

# Calculate accuracy, precision, recall, F1
accuracy = (true_pos+true_neg) / n_pixels
precision = true_pos / (true_pos+false_pos)
recall = true_pos / (true_pos+false_neg)
F1 = 2 * ((precision*recall)/(precision+recall))
print('Total number of pixels ={} :'.format(n_pixels))
print(' - # true positives = {:6.0f}'.format( true_pos ))
print(' - # false positives = {:6.0f}'.format( false_pos ))
print(' - # false negatives = {:6.0f}'.format( false_neg ))
print(' - # true negatives = {:6.0f}'.format( true_neg ))
print(' - Accuracy = {:6.4f}'.format( accuracy ))
print(' - Precision = {:6.4f}'.format( precision ))
print(' - Recall = {:6.4f}'.format( recall ))
print(' - F1-score = {:6.4f}'.format( F1 ))


########################################################################
### Calculate annotation based performance
print("\n-------- Annotation based performance --------")

print('Ground truth # annotations = {:4d}'.format( len(anim_gt.annotation) ))
print('Neural net # annotations = {:4d}'.format( len(anim_nn.annotation) ))

# Predefine annotation match lists
gt_nn_list = [[] for _ in anim_gt.annotation]
nn_gt_list = [[] for _ in anim_nn.annotation]
gt_nn_body_list = [[] for _ in anim_gt.annotation]
nn_gt_body_list = [[] for _ in anim_nn.annotation]

# Find putative overlapping annotations by 'centroid_distance'
ctrd_true_pos = 0
ctrd_true_pos_list_gt = []
ctrd_true_pos_list_nn = []
print("\nSearching nearby annotation centroids (<{:3.0f} micron) {:3d}".format( \
        max_centroid_distance, 0), end="", flush=True)
for nr_gt,an_gt in enumerate(anim_gt.annotation):
    print((3*'\b')+'{:3d}'.format(nr_gt+1), end='', flush=True)
    for nr_nn,an_nn in enumerate(anim_nn.annotation):
        D = np.sqrt( ((an_gt.y-an_nn.y)**2) + ((an_gt.x-an_nn.x)**2) )
        if D < max_centroid_distance:
            ctrd_true_pos += 1
            ctrd_true_pos_list_gt.append(nr_gt)
            ctrd_true_pos_list_nn.append(nr_nn)
        if D < max_body_centroid_distance:
            gt_nn_list[nr_gt].append(nr_nn)
            nn_gt_list[nr_nn].append(nr_gt)
print((3*'\b')+'{:3d}'.format(nr_gt+1))

ctrd_false_pos = len(anim_nn.annotation)-ctrd_true_pos
ctrd_false_neg = len(anim_gt.annotation)-ctrd_true_pos
ctrd_precision = ctrd_true_pos / (ctrd_true_pos+ctrd_false_pos)
ctrd_recall = ctrd_true_pos / (ctrd_true_pos+ctrd_false_neg)
ctrd_F1 = 2 * ((ctrd_precision*ctrd_recall)/(ctrd_precision+ctrd_recall))

print(' - # true positives = {:6.0f}'.format( ctrd_true_pos ))
print(' - # false positives = {:6.0f}'.format( ctrd_false_pos ))
print(' - # false negatives = {:6.0f}'.format( ctrd_false_neg ))
print(' - Precision = {:6.4f}'.format( ctrd_precision ))
print(' - Recall = {:6.4f}'.format( ctrd_recall ))
print(' - F1-score = {:6.4f}'.format( ctrd_F1 ))


# Find annotations with > 'overlap_fraction' overlapping area
print("\nSearching for GT to NN body overlap (>{:3.0f}% area) {:3d}".format( \
        min_body_overlap_fraction*100, 0), end="", flush=True)
for nr_gt,nrs_nn in enumerate(gt_nn_list):
    print((3*'\b')+'{:3d}'.format(nr_gt+1), end='', flush=True)
    mask_gt = anim_gt.bodies == nr_gt+1
    for nr_nn in nrs_nn:
        mask_nn = anim_nn.bodies == nr_nn+1
        if np.sum(mask_gt[mask_nn]) > (min_body_overlap_fraction*mask_gt.sum()):
            gt_nn_body_list[nr_gt].append(nr_nn)
print((3*'\b')+'{:3d}'.format(nr_gt+1))

print("Searching for NN to GT body overlap (>{:3.0f}% area) {:3d}".format( \
        min_body_overlap_fraction*100, 0), end="", flush=True)
for nr_nn,nrs_gt in enumerate(nn_gt_list):
    print((3*'\b')+'{:3d}'.format(nr_nn+1), end='', flush=True)
    mask_nn = anim_nn.bodies == nr_nn+1
    for nr_gt in nrs_gt:
        mask_gt = anim_gt.bodies == nr_gt+1
        if np.sum(mask_nn[mask_gt]) > (min_body_overlap_fraction*mask_nn.sum()):
            nn_gt_body_list[nr_nn].append(nr_gt)
print((3*'\b')+'{:3d}'.format(nr_nn+1))

body_true_pos = 0
body_true_pos_list_gt = []
body_true_pos_list_nn = []
for nr_gt,nrs_nn in enumerate(gt_nn_body_list):
    if len(nrs_nn) == 1:
        if len(nn_gt_body_list[nrs_nn[0]]) == 1:
            if nn_gt_body_list[nrs_nn[0]][0] == nr_gt:
                # Both the NN and GT annotation overlap 'overlap_fraction'
                # with eachother and not with others
                body_true_pos += 1
                body_true_pos_list_gt.append(nr_gt)
                body_true_pos_list_nn.append(nrs_nn[0])

print("Bidirectional overlap (>{:3.0f}% area):".format( \
        min_body_overlap_fraction*100, 0) )

body_false_pos = len(anim_nn.annotation)-body_true_pos
body_false_neg = len(anim_gt.annotation)-body_true_pos
body_precision = body_true_pos / (body_true_pos+body_false_pos)
body_recall = body_true_pos / (body_true_pos+body_false_neg)
body_F1 = 2 * ((body_precision*body_recall)/(body_precision+body_recall))

print(' - # true positives = {:6.0f}'.format( body_true_pos ))
print(' - # false positives = {:6.0f}'.format( body_false_pos ))
print(' - # false negatives = {:6.0f}'.format( body_false_neg ))
print(' - Precision = {:6.4f}'.format( body_precision ))
print(' - Recall = {:6.4f}'.format( body_recall ))
print(' - F1-score = {:6.4f}'.format( body_F1 ))


# ************************************************************
# Show matplotlib images
anim_gt.centroid_dilation_factor = 2
RGB_gt = np.zeros((anim_gt.bodies.shape[0],anim_gt.bodies.shape[1],3))
RGB_gt[:,:,1] = anim_gt.centroids>0
RGB_gt[:,:,2] = anim_gt.bodies>0

RGB_nn_raw = np.zeros((anim_nn.detected_bodies.shape[0],anim_nn.detected_bodies.shape[1],3))
if anim_nn.detected_centroids is not None:
    RGB_nn_raw[:,:,1] = anim_nn.detected_centroids
RGB_nn_raw[:,:,2] = anim_nn.detected_bodies

anim_nn.centroid_dilation_factor = 2
RGB_nn = np.zeros((anim_nn.bodies.shape[0],anim_nn.bodies.shape[1],3))
RGB_nn[:,:,1] = anim_nn.centroids>0
RGB_nn[:,:,2] = anim_nn.bodies>0

# Show image and classification result
with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax = plt.subplot2grid( (1,3), (0,0) )
    ax.imshow(RGB_gt)
    ax.set_title("Ground truth")
    plt.axis('tight')
    plt.axis('off')

    ax = plt.subplot2grid( (1,3), (0,1) )
    ax.imshow(RGB_nn)
    ax.set_title("Neural Net")
    plt.axis('tight')
    plt.axis('off')

    ax = plt.subplot2grid( (1,3), (0,2) )
    ax.imshow(RGB_nn_raw)
    ax.set_title("Neural Net, raw output")
    plt.axis('tight')
    plt.axis('off')

with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax = plt.subplot2grid( (1,2), (0,0) )
    ax.imshow(RGB_gt)
    for nr,an in enumerate(anim_nn.annotation):
        if nr in ctrd_true_pos_list_nn:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ff0000" )
        else:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ffffff" )
    ax.set_title("Ground truth + Neural Net annotations (red=centroid true-pos)")
    plt.axis('tight')
    plt.axis('off')

    ax = plt.subplot2grid( (1,2), (0,1) )
    ax.imshow(RGB_nn)
    for nr,an in enumerate(anim_gt.annotation):
        if nr in ctrd_true_pos_list_gt:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ff0000" )
        else:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ffffff" )
    ax.set_title("Neural Net + Ground Truth annotations (red=centroid true-pos)")
    plt.axis('tight')
    plt.axis('off')

with sns.axes_style("white"):
    plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
    ax = plt.subplot2grid( (1,2), (0,0) )
    ax.imshow(RGB_gt)
    for nr,an in enumerate(anim_nn.annotation):
        if nr in body_true_pos_list_nn:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ff0000" )
        else:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ffffff" )
    ax.set_title("Ground truth + Neural Net annotations (red=body true-pos)")
    plt.axis('tight')
    plt.axis('off')

    ax = plt.subplot2grid( (1,2), (0,1) )
    ax.imshow(RGB_nn)
    for nr,an in enumerate(anim_gt.annotation):
        if nr in body_true_pos_list_gt:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ff0000" )
        else:
            ax.plot( an.perimeter[:,1], an.perimeter[:,0],
                linewidth=1, color="#ffffff" )
    ax.set_title("Neural Net + Ground Truth annotations (red=body true-pos)")
    plt.axis('tight')
    plt.axis('off')

print('\nDone!\n')
plt.show()
