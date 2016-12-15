#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:25:32 2016

Contains functions that represent (sets of) images and their annotations

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
from skimage import measure
from scipy import ndimage


########################################################################
### Class Annotation
########################################################################

class Annotation(object):
    """Class that holds individual image annotations"""

    def __init__(self,body_pixels_yx,annotation_name,group_nr=None):
        """Initialize.
            body_pixels_yx: list/tuple of (y,x)'s or [y,x]'s
            annotation_name: string
            group_nr: int
        """
        # Store supplied parameters
        self.body = body_pixels_yx
        self.name = annotation_name

    def __str__(self):
        return "Annotation at (y={:.1f},x={:.1f}), name={!s}".format(
            self.__y,self.__x,self.name)

    @property
    def body(self):
        """Returns body coordinates"""
        return self.__body

    @body.setter
    def body(self,body_pixels_yx):
        """Sets body coordinates and calculates associated centroids"""
        self.__body = np.array(body_pixels_yx)
        self.__y = self.__body[:,0].mean()
        self.__x = self.__body[:,1].mean()
        temp_mask = np.zeros( self.__body.max(axis=0)+1 )
        temp_mask[ np.ix_(self.__body[:,0],self.__body[:,1]) ] = 1
        self.__perimeter = measure.find_contours(temp_mask, 0.5)[0]
        self.__size = self.__body.shape[0]

    @property
    def x(self):
        """Returns read-only centroid x coordinate"""
        return self.__x

    @property
    def y(self):
        """Returns read-only centroid y coordinate"""
        return self.__y

    @property
    def perimeter(self):
        """Returns read-only stored list of perimeter (y,x) coordinates"""
        return self.__perimeter

    @property
    def size(self):
        """Returns read-only size of annotation (number of pixels)"""
        return self.__size

    def mask_body(self, image, dilation_factor=0, mask_value=1):
        """Draws mask of all body pixels in image
        dilation_factor: >0 for dilation, <0 for erosion"""
        if dilation_factor==0:
            # Just mask the incoming image
            image[ np.ix_(self.__body[:,0],self.__body[:,1]) ] = mask_value
        else:
            # Draw mask on temp image, dilate, get pixels, then draw in image
            temp_mask = np.zeros_like(image,dtype=bool)
            temp_mask[ np.ix_(self.__body[:,0],self.__body[:,1]) ] = True
            if dilation_factor>0:
                for _ in range(dilation_factor):
                    temp_mask = ndimage.binary_dilation(temp_mask)
            elif dilation_factor<0:
                for _ in range(-1*dilation_factor):
                    temp_mask = ndimage.binary_erosion(temp_mask)
            temp_body = np.array(np.where(temp_mask == True)).transpose()
            image[ np.ix_(temp_body[:,0],temp_body[:,1]) ] = mask_value

    def mask_centroid(self, image, dilation_factor=0, mask_value=1):
        """Draws mask of centroid pixel in image
        dilation_factor: >0 for padding the centroid with surrounding points"""
        if dilation_factor==0:
            # Just mask the incoming image
            image[self.__y.astype(int),self.__x.astype(int)] = mask_value
        else:
            # Draw mask on temp image, dilate, get pixels, then draw in image
            temp_mask = np.zeros_like(image,dtype=bool)
            temp_mask[self.__y.astype(int),self.__x.astype(int)] = True
            for _ in range(dilation_factor):
                temp_mask = ndimage.binary_dilation(temp_mask)
            temp_body = np.array(np.where(temp_mask == True)).transpose()
            image[ np.ix_(temp_body[:,0],temp_body[:,1]) ] = mask_value

    def zoom(self,image,zoom_size):
        """Crops image to area of tuple/list zoom_size around centroid"""
        top_y  = np.int16( 1 + self.__y - (zoom_size[0] / 2) )
        left_x = np.int16( 1 + self.__x - (zoom_size[1] / 2) )
        ix_y = top_y + list(range( 0, zoom_size[0] ))
        ix_x = left_x + list(range( 0, zoom_size[1] ))
        return image[ np.ix_(ix_y,ix_x) ]


########################################################################
### Class AnnotatedImage
########################################################################

class AnnotatedImage(object):
    """Class that hold a multichannel image and its annotations
    Images are represented in a [x * y * nChannels] matrix
    Annotations are represented as a list of Annotation objects"""

    def __init__(self,image_size):
        self.y_res,self.x_res = image_size
        self.image = np.zeros(image_size)
        self.annotation_list = []

    def import_from_mat(self,file_name,file_path='.'):
        """Reads data from ROI.mat file and fills the annotation_list"""
        return 0

    def export_annotations_to_mat(self,file_name,file_path='.'):
        """Writes annotations to ROI_py.mat file"""
        return 0

    def load(self,file_name,file_path='.'):
        """Loads image and annotations from file"""

    def save(self,file_name,file_path='.'):
        """Saves image and annotations to file"""


########################################################################
### Class AnnotatedImageSet
########################################################################

class AnnotatedImageSet(object):
    """Class that represents a dataset of annotated images and organizes
    the dataset for feeding in machine learning algorithms"""

    def __init__(self):
        print("Class not yet implemented...")

    def import_from_mat(self,data_directory):
        """Imports all images and accompanying ROI.mat files from a
        single directory"""

    def get_training_set():
        """Returns training set"""
        return 0

    def get_crossvalidation_set():
        """Returns cross-validation set"""
        return 0

    def get_test_set():
        """Returns test set"""
        return 0

    def get_full_set():
        """Returns the entire data set (train, cv, test)"""
        return 0
