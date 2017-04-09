#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:08:21 2016

@author: pgoltstein
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print(" ")
print("----------------------------------------------------------")
print("Importing ImageAnnotation as ia")
import ImageAnnotation as ia

# Define a single annotation to test with
print("Define a simple annotation named 'funky2'")
a1 = ia.Annotation([[10,10],[10,11],[10,12],
                   [11,10],[11,11],[11,12],
                   [12,10],[12,11],[12,12],
                   [13,10],[13,11],[13,12],
                   [14,10],[14,11],[14,12]],'funky2')
print("String output of the annotation:")
print(" >> "+a1.__str__())

# Create a mask of the annotation
print(" ")
print("Mask body(val=1) and centroid(val=2) in a 20x20 zero-valued image")
im1=np.zeros((20,20))
a1.mask_body(im1,dilation_factor=0,mask_value=1)
a1.mask_centroid(im1,dilation_factor=0,mask_value=2)

# Create a zoom of the annotation, and create a new annotation based on that
zoom_size = 11
im2 = a1.zoom(image=im1,zoom_size=(zoom_size,zoom_size))
a2 = ia.Annotation(im2>0.5)
print(" ")
print("Zoom-in (11x11 pixels) of the annotation")
print("Center coordinate: {},{}".format( np.argmax(np.max(im2,axis=0)),
      np.argmax(np.max(im2,axis=1))) )
print(" >> "+a1.__str__())

# Create a morphed zoom of the annotation, and create a new annotation
im3 = a1.morphed_zoom(image=im1, zoom_size=(zoom_size,zoom_size),
                     rotation=45, scale_xy=(0.9,1.3), noise_level=0.0)
a3 = ia.Annotation(im3>0.5)
print(" ")
print("Morphed zoom-in of the annotation")
print("Center coordinate: {},{}".format( np.argmax(np.max(im3,axis=0)),
      np.argmax(np.max(im3,axis=1))) )
print(" >> "+a1.__str__())

# Create some copies of annotion a1 and a3 and draw them in an image
print(" ")
print("Five copies of annotations")
a = []
a.append(ia.Annotation(a1.body))
a.append(ia.Annotation(a1.body+20))
a.append(
    ia.Annotation(np.transpose(np.array([a1.body[:,0]+5,a1.body[:,1]+30]))))
a.append(ia.Annotation(a2.body+20))
a.append(
    ia.Annotation(np.transpose(np.array([a3.body[:,0]+35,a3.body[:,1]+20]))))
im4=np.zeros((60,60))
for an in a:
    an.mask_body(im4,dilation_factor=0,mask_value=1)
    an.mask_centroid(im4,dilation_factor=0,mask_value=2)
    print(" >> "+an.__str__())

# ************************************************************
# Show matplotlib images

with sns.axes_style("white"):
    # Display masked annotation a1 and its outline
    plt.figure(figsize=(8,8), facecolor='w', edgecolor='w')
    ax1 = plt.subplot2grid( (2,2), (0,0) )
    ax1.imshow(im1,interpolation='nearest')
    ax1.plot( a1.perimeter[:,1], a1.perimeter[:,0], color="#aa00aa" )
    ax1.set_title("Test annotation 'a1'")
    plt.axis('tight')

    ax2 = plt.subplot2grid( (2,2), (0,1) )
    ax2.imshow(im2,interpolation='nearest')
    ax2.plot( a2.perimeter[:,1], a2.perimeter[:,0], color="#aa00aa" )
    ax2.set_title("Zoom-in of 'a1', annotated as 'a2'")
    plt.axis('tight')

    ax3 = plt.subplot2grid( (2,2), (1,0) )
    ax3.imshow(im3,interpolation='nearest')
    ax3.plot( a3.perimeter[:,1], a3.perimeter[:,0], color="#aa00aa" )
    ax3.set_title("Morphed zoom-in of 'a1', annotated as 'a3'")
    plt.axis('tight')

    ax4 = plt.subplot2grid( (2,2), (1,1) )
    ax4.imshow(im4,interpolation='nearest')
    for an in a:
        ax4.plot( an.perimeter[:,1], an.perimeter[:,0], color="#aa00aa" )
    ax4.set_title("Five copies of annotations")
    plt.axis('tight')

# Show plots
# sns.set_context("talk")
plt.show()
