#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 17:49:54 2017

Contains functions that detect centroids of annotations

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
import tensorflow as tf
import ImageAnnotation as ia
import iaConvNetTools as cn


########################################################################
# Load data

data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet1'
zoom_size = (36,36)
training_data = ia.AnnotatedImageSet()
training_data.load_data_dir(data_path)

########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1( input_image_size=zoom_size, n_input_channels=3,
                        output_size=(1,2) )
nn.start()

########################################################################
# Train network
nn.train( training_data, m_samples=500, n_epochs=20, display_every_n=4)
