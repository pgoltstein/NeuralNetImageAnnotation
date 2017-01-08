#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:38:41 2017

@author: pgoltstein
"""

import ImageAnnotation as ia
import numpy as np
import importlib
import matplotlib.pyplot as plt
from skimage import measure

importlib.reload(ia)

data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
ais = ia.AnnotatedImageSet()
ais.load_data_dir(data_path)
sam,lab = ais.centroid_detection_data(m_samples=100)
print(sam.shape)
print(lab.shape)
for t in range(100):print(lab[t])

#
#ai_list = ais.full_data_set


an_im_nr = 1
im_ch_nr = 0
ann_nr = 24

[im,zm] = ai_list[an_im_nr].annotation[ann_nr].morphed_zoom( 
            ai_list[an_im_nr].channel[im_ch_nr],rotation=00)
perimeter = measure.find_contours( zm, 0.5 )[0]
_, ax = plt.subplots(1,2)

ax[0].imshow(im)
ax[0].plot(perimeter[:,1],perimeter[:,0],'w')
ax[0].set_title('Channel {}'.format(im_ch_nr))

ax[1].imshow(zm)
ax[1].set_title('Annotation')
