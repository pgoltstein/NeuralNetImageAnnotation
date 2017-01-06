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

importlib.reload(ia)

anim = ia.AnnotatedImage()
print(anim)

filepath = '/Users/pgoltstein/Dropbox/TEMP'
roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'

anim.import_annotations_from_mat(file_name=roifile,file_path=filepath)
print(anim)

anim.add_image_from_file(file_name=imfile,file_path=filepath)
print(anim)

plt.figure()
plt.imshow(anim.channel[0],interpolation='nearest')

plt.figure()
plt.imshow(anim.bodies(),interpolation='nearest')

image = anim.channel[0]
image[anim.centroids(dilation_factor=5)==1]=0
plt.figure()
plt.imshow(image,interpolation='nearest')


