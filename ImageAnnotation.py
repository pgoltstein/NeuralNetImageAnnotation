#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:25:32 2016

Contains functions that represent (sets of) images and their annotations

@author: pgoltstein
"""


class Annotation(object):
    """Class that holds individual annotation data"""

    def __init__(self,body_pixels_yx,annotation_type):
        # Store body pixels
        self.body = body_pixels_yx

        # Calculate centroids
        self.y = np.mean(self.body[:,0])
        self.x = np.mean(self.body[:,1])

    @property
    def perimeter():
        return []


class AnnotatedImage(object):
    """Class that hold a multichannel image and its annotations
    Images are represented in a [x * y * nChannels] matrix
    Annotations are represented as a list of Annotation objects"""

    def __init__(self,image_size):
        self.y_res,self.x_res = image_size
        self.I = np.zeros(image_size)

    def import_from_mat(file_name,file_path='.'):
        return 0


class ML_ROI_Set(object):
    """Class that represents a dataset of annotated images and organizes
    the dataset for feeding in machine learning algorithms"""

    def __init__(self):
        print("Function not yet implemented...")
