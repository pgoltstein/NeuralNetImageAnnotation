#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:25:32 2016

Contains functions that represent (sets of) images and their annotations

1. class Annotation(object):
    Class that holds an individual image annotation

2. class AnnotatedImage(object):
    Class that hold a multichannel image and its annotations
    Images are represented in a [x * y * n_channels] matrix
    Annotations are represented as a list of Annotation objects

3. class AnnotatedImageSet(object):
    Class that represents a dataset of annotated images and organizes
    the dataset for feeding in machine learning algorithms

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
from skimage import measure
from skimage.io import imread
from scipy import ndimage
from scipy.io import loadmat
from os import path


########################################################################
### Class Annotation
########################################################################

class Annotation(object):
    """Class that holds an individual image annotation"""

    DEFAULT_ZOOM = (27,27)

    def __init__(self,body_pixels_yx,annotation_name,group_nr=None):
        """Initialize.
            body_pixels_yx: list/tuple of (y,x)'s or [y,x]'s
            annotation_name: string
            group_nr: int
        """
        # Store supplied parameters
        self.body = np.int16(body_pixels_yx)
        self.name = annotation_name
        self.group_nr = group_nr

    def __str__(self):
        return "Annotation at (y={:.1f},x={:.1f}), group={:.0f}, "\
            "name={!s}".format(self._y, self._x, self._group_nr, self.name)

    @property
    def body(self):
        """Returns body coordinates"""
        return self._body

    @body.setter
    def body(self,body_pixels_yx):
        """Sets body coordinates and calculates associated centroids"""
        self._body = np.array(body_pixels_yx)
        self._y = self._body[:,0].mean()
        self._x = self._body[:,1].mean()
        temp_mask = np.zeros( self._body.max(axis=0)+1 )
        temp_mask[ np.ix_(self._body[:,0],self._body[:,1]) ] = 1
        self._perimeter = measure.find_contours(temp_mask, 0.5)[0]
        self._size = self._body.shape[0]

    @property
    def x(self):
        """Returns read-only centroid x coordinate"""
        return self._x

    @property
    def y(self):
        """Returns read-only centroid y coordinate"""
        return self._y

    @property
    def group_nr(self):
        """Returns read-only group number"""
        return self._group_nr

    @group_nr.setter
    def group_nr(self,group_nr):
        """Sets group number to integer or 0"""
        if isinstance(group_nr, (int,float)):
            self._group_nr = int(group_nr)
        else:
            self._group_nr = 0

    @property
    def perimeter(self):
        """Returns read-only stored list of perimeter (y,x) coordinates"""
        return self._perimeter

    @property
    def size(self):
        """Returns read-only size of annotation (number of pixels)"""
        return self._size

    def mask_body(self, image, dilation_factor=0, mask_value=1):
        """Draws mask of all body pixels in image
        dilation_factor: >0 for dilation, <0 for erosion"""
        if dilation_factor==0:
            # Just mask the incoming image
            image[ self._body[:,0], self._body[:,1] ] = mask_value
        else:
            # Draw mask on temp image, dilate, get pixels, then draw in image
            temp_mask = np.zeros_like(image,dtype=bool)
            temp_mask[ np.ix_(self._body[:,0],self._body[:,1]) ] = True
            if dilation_factor>0:
                for _ in range(dilation_factor):
                    temp_mask = ndimage.binary_dilation(temp_mask)
            elif dilation_factor<0:
                for _ in range(-1*dilation_factor):
                    temp_mask = ndimage.binary_erosion(temp_mask)
            temp_body = np.array(np.where(temp_mask == True)).transpose()
            image[ temp_body[:,0], temp_body[:,1] ] = mask_value

    def mask_centroid(self, image, dilation_factor=0, mask_value=1):
        """Draws mask of centroid pixel in image
        dilation_factor: >0 for padding the centroid with surrounding points"""
        if dilation_factor==0:
            # Just mask the incoming image
            image[self._y.astype(int),self._x.astype(int)] = mask_value
        else:
            # Draw mask on temp image, dilate, get pixels, then draw in image
            temp_mask = np.zeros_like(image,dtype=bool)
            temp_mask[self._y.astype(int),self._x.astype(int)] = True
            for _ in range(dilation_factor):
                temp_mask = ndimage.binary_dilation(temp_mask)
            temp_body = np.array(np.where(temp_mask == True)).transpose()
            image[ temp_body[:,0], temp_body[:,1] ] = mask_value

    def zoom(self, image, zoom_size=DEFAULT_ZOOM ):
        """Crops image to area of tuple/list zoom_size around centroid
        zoom_size: (y size, x size), accepts only uneven numbers"""
        assert zoom_size[0] % 2 and zoom_size[1] % 2, \
            "zoom_size cannot contain even numbers: (%r,%r)" % zoom_size
        top_y  = np.int16( 1 + self._y - ((zoom_size[0]+1) / 2) )
        left_x = np.int16( 1 + self._x - ((zoom_size[1]+1) / 2) )
        ix_y = top_y + list(range( 0, zoom_size[0] ))
        ix_x = left_x + list(range( 0, zoom_size[1] ))
        return image[ np.ix_(ix_y,ix_x) ]

    def morped_zoom(self, image, zoom_size=DEFAULT_ZOOM, rotation=0,
                    scale_xy=(1,1), noise_level=0 ):
        """Crops image to area of tuple/list zoom_size around centroid
        zoom_size:   (y size, x size), accepts only uneven numbers
        rotation:    Rotation of annotation in degrees (0-360 degrees)
        scale_xy:    Determines fractional scaling on x/y axis.
                     Min-Max = (0.5,0.5) - (2,2)
        noise_level: Level of random noise
        returns tuple holding (morped_zoom, morped_annotation)"""

        assert zoom_size[0] % 2 and zoom_size[1] % 2, \
            "zoom_size cannot contain even numbers: (%r,%r)" % zoom_size

        # Get large, annotation centered, zoom image
        temp_zoom_size = (zoom_size[0]*4+1,zoom_size[1]*4+1)
        temp_zoom = self.zoom(image, temp_zoom_size )

        # Get large, annotation centered, mask image
        mid_y = np.int16(temp_zoom_size[0] / 2)
        mid_x = np.int16(temp_zoom_size[1] / 2)
        top_y  = np.int16( 1 + mid_y - (zoom_size[0] / 2) )
        left_x = np.int16( 1 + mid_x - (zoom_size[1] / 2) )
        temp_ann = np.zeros_like(temp_zoom)
        temp_ann[ np.int16((self._body[:,0]-self._y) + zoom_size[0]*2),
                  np.int16((self._body[:,1]-self._x) + zoom_size[0]*2) ] = 1

        # Rotate
        if rotation != 0:
            temp_zoom = ndimage.interpolation.rotate(temp_zoom,
                            rotation, reshape=False)
            temp_ann = ndimage.interpolation.rotate(temp_ann,
                            rotation, reshape=False)

        # Scale
        if scale_xy[0] != 1 or scale_xy[1] != 1:
            temp_zoom = ndimage.interpolation.zoom( temp_zoom, scale_xy )
            temp_ann = ndimage.interpolation.zoom( temp_ann, scale_xy )
            temp_zoom_size = temp_zoom.shape

        # Add noise
        if noise_level:
            noise_mask = np.random.normal(size=temp_zoom.shape) * noise_level
            temp_zoom = temp_zoom + noise_mask

        # Make mask 0 or 1
        temp_ann[temp_ann<0.5] = 0
        temp_ann[temp_ann>=0.5] = 1

        # Cut out real zoom image from center of temp_zoom
        mid_y = np.int16(temp_zoom_size[0] / 2)
        mid_x = np.int16(temp_zoom_size[1] / 2)
        top_y  = np.int16( 1 + mid_y - (zoom_size[0] / 2) )
        left_x = np.int16( 1 + mid_x - (zoom_size[1] / 2) )
        ix_y = top_y + list(range( 0, zoom_size[0] ))
        ix_x = left_x + list(range( 0, zoom_size[1] ))
        return (temp_zoom[ np.ix_(ix_y,ix_x) ],temp_ann[ np.ix_(ix_y,ix_x) ])


########################################################################
### Class AnnotatedImage
########################################################################


class AnnotatedImage(object):
    """Class that hold a multichannel image and its annotations
    Images are represented in a list of [x * y] matrices
    Annotations are represented as a list of Annotation objects
    channel      = list or tuple of same size images
    annotation = list or tuple of Annotation objects"""

    def __init__( self, image_data=[], annotation_data=[] ):
        self.channel = image_data
        self.annotation = annotation_data

    def __str__(self):
        return "AnnotatedImage (n_channels={:.0f}, n_annotations={:.0f}" \
                ")".format(self.n_channels, self.n_annotations)

    @property
    def n_channels(self):
        return len(self.channel)

    @property
    def n_annotations(self):
        return len(self.annotation)

    def add_image_from_file(self,file_name,file_path='.'):
        """Loads image and adds it as a new extra channel"""
        im = np.float64(imread(path.join(file_path,file_name)))
        if im.ndim == 3:
            n_channels = np.size(im,axis=2)
            for ch in range(n_channels):
                self.channel.append(im[:,:,ch])
        else:
            self.channel.append(im)

    def import_annotations_from_mat(self,file_name,file_path='.'):
        """Reads data from ROI.mat file and fills the annotation_list"""
        # Load mat file with ROI data
        mat_data = loadmat(path.join(file_path,file_name))
        nROIs = len(mat_data['ROI'][0])
        for c in range(nROIs):
            body = mat_data['ROI'][0][c][8]
            body = np.array([body[:,1],body[:,0]]).transpose()
            name = mat_data['ROI'][0][c][3][0]
            group_nr = mat_data['ROI'][0][c][1][0][0]
            self.annotation.append( Annotation( body_pixels_yx=body,
                                    annotation_name=name, group_nr=group_nr ) )

    def bodies(self, dilation_factor=0, mask_value=1):
        annotated_bodies = np.zeros_like(self.channel[0])
        for nr in range(self.n_annotations):
            self.annotation[nr].mask_body(annotated_bodies,
                        dilation_factor=dilation_factor, mask_value=mask_value)
        return annotated_bodies

    def centroids(self, dilation_factor=0, mask_value=1):
        annotated_centroids = np.zeros_like(self.channel[0])
        for nr in range(self.n_annotations):
            self.annotation[nr].mask_centroid(annotated_centroids,
                        dilation_factor=dilation_factor, mask_value=mask_value)
        return annotated_centroids

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
        self._an_im_list = []
        print("Class unfinished...")

    def __str__(self):
        return "AnnotatedImageSet (n_sets={:.0f}" \
                ")".format(self._an_im_list)

    @property
    def n_sets(self):
        return len(self.channel)

    @property
    def n_annotations(self):
        return len(self.annotation)

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
