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
from skimage import measure

print(" ")
print("----------------------------------------------------------")
print("Importing ImageAnnotation as ia")
import ImageAnnotation as ia

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Overall settings
filepath = '/Users/pgoltstein/Dropbox/TEMP/DataSet_small'
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
print("data_path")
ais.load_data_dir(data_path)
print(" >> " + ais1.__str__())

# sam,lab = ais.centroid_detection_data(m_samples=100)
# print(sam.shape)
# print(lab.shape)
# for t in range(100):print(lab[t])
#
# #
# #ai_list = ais.full_data_set
#
#
# an_im_nr = 1
# im_ch_nr = 0
# ann_nr = 24
#
# [im,zm] = ai_list[an_im_nr].annotation[ann_nr].morphed_zoom(
#             ai_list[an_im_nr].channel[im_ch_nr],rotation=00)
# perimeter = measure.find_contours( zm, 0.5 )[0]
# _, ax = plt.subplots(1,2)
#
# ax[0].imshow(im)
# ax[0].plot(perimeter[:,1],perimeter[:,0],'w')
# ax[0].set_title('Channel {}'.format(im_ch_nr))
#
# ax[1].imshow(zm)
# ax[1].set_title('Annotation')
