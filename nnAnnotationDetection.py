#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 10 2017

Contains functions that detect bodies of annotations in a single step

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import ImageAnnotation as ia
import iaConvNetTools as cn
import os

########################################################################
# Settings and variables
annotation_size = (27,27)
data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet2'
network_path = '/Users/pgoltstein/Dropbox/TEMP'
roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'

########################################################################
# Load data
print("\nLoading data into AnnotatedImage classes named: anim1 and anim2")
anim1 = ia.AnnotatedImage()
anim1.import_annotations_from_mat(file_name=roifile,file_path=data_path)
anim1.add_image_from_file(file_name=imfile,file_path=data_path)
print(" anim1 >> " + anim1.__str__())
anim2 = ia.AnnotatedImage()
anim2.add_image_from_file(file_name=imfile,file_path=data_path)
print(" anim2 >> " + anim2.__str__())

########################################################################
# Set up network
print("Setting up convolutional neural network")
nn = cn.ConvNetCnv2Fc1( \
        input_image_size=annotation_size,
        n_input_channels=anim1.n_channels, output_size=(1,2),
        conv1_size=7, conv1_n_chan=32, conv1_n_pool=3,
        conv2_size=7, conv2_n_chan=64, conv2_n_pool=3,
        fc1_n_chan=256, fc1_dropout=0.5, alpha=4e-4 )
nn.start()
nn.load_network_parameters('body1step_net',network_path)

########################################################################
# Annotate image
print("Running classification on anim2")
classified_image = nn.annotate_image( anim2 )

# Save classified image
np.save(os.path.join(data_path,'classified_image'), classified_image)
print("Saved classified_image as: {}".format(
                            path.join(data_path,'classified_image')))

# ************************************************************
# Show matplotlib images

# Show image and classification result
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
    axb.imshow(anim1.bodies>0,vmin=-0.1)
    axb.set_title("anim: Annotated bodies")
    plt.axis('tight')
    plt.axis('off')

    axb = plt.subplot2grid( (2,3), (1,1) )
    axb.imshow(classified_image,vmin=-0.1)
    axb.set_title("anim2: Classification result")
    plt.axis('tight')
    plt.axis('off')

    axr = plt.subplot2grid( (2,3), (1,2) )
    axr.imshow( anim1.RGB(),
        interpolation='nearest', vmax=anim1.RGB().max()*0.7)
    for an in anim1.annotation:
        axr.plot( an.perimeter[:,1], an.perimeter[:,0],
            linewidth=1, color="#ffffff" )
    axr.set_title("anim: RGB with annotations")
    plt.axis('tight')
    plt.axis('off')

print('Done!\n')
plt.show()
