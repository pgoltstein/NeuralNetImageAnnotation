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

########################################################################
### Load data
########################################################################

data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet'
ais = ia.AnnotatedImageSet()
ais.load_data_dir(data_path)
