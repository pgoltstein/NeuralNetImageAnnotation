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

filepath = '/Users/pgoltstein/Dropbox/TEMP'
roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'

#anim.import_annotations_from_mat(file_name=roifile,file_path=filepath)
anim.import_annotations_from_mat(file_name='0test_rois.mat',file_path=filepath)
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

#anim.export_annotations_to_mat(file_name='0test_rois',file_path=filepath)

#mat_data = loadmat(os.path.join(filepath,'0test_rois'))
#mat_data2 = loadmat(os.path.join(filepath,roifile))


anim2 = ia.AnnotatedImage()
print(anim2)

filepath = '/Users/pgoltstein/Dropbox/TEMP'
roifile = 'F03-Loc5-V1-20160209-ROI2.mat'
imfile = 'F03-Loc5-V1-20160209-OverlayL2.tiff'

anim2.import_annotations_from_mat(file_name=roifile,file_path=filepath)


