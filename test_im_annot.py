#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:08:21 2016

@author: pgoltstein
"""

import ImageAnnotation as ia
import numpy as np
import importlib
import matplotlib.pyplot as plt

importlib.reload(ia)

a = ia.Annotation([[10,10],[10,11],[10,12],
                   [11,10],[11,11],[11,12],
                   [12,10],[12,11],[12,12]],'funky2')

b=np.zeros((100,100))

a.mask_body(b,dilation_factor=0,mask_value=3)
a.mask_centroid(b,dilation_factor=0,mask_value=1)


plt.figure()
plt.imshow(b,interpolation='nearest')
print(a)

zoom_size = 36

c = a.zoom(b,(zoom_size,zoom_size))
plt.figure()
plt.imshow(c,interpolation='nearest')

c,d = a.morphed_zoom(b,zoom_size=(zoom_size,zoom_size),rotation=45, scale_xy=(1,2), noise_level=0.1)
plt.figure()
plt.imshow(c,interpolation='nearest')
plt.figure()
plt.imshow(d,interpolation='nearest')
