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
from scipy.io import loadmat,savemat
from os import path
import glob


########################################################################
### Constants
########################################################################

DEFAULT_ZOOM = (33,33)


########################################################################
### Class Annotation
########################################################################

class Annotation(object):
    """Class that holds an individual image annotation"""

    def __init__(self,body_pixels_yx,annotation_name,type_nr=1,group_nr=None):
        """Initialize.
            body_pixels_yx: list/tuple of (y,x)'s or [y,x]'s
            annotation_name: string
            type_nr: int
            group_nr: int
        """
        # Store supplied parameters
        self.body = np.int16(body_pixels_yx)
        self.name = annotation_name
        self.type_nr = type_nr
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
        temp_mask[ self._body[:,0], self._body[:,1] ] = 1
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
    def type_nr(self):
        """Returns read-only type number"""
        return self._type_nr

    @type_nr.setter
    def type_nr(self,type_nr):
        """Sets type number to integer or 0"""
        if isinstance(type_nr, (int,float)):
            self._type_nr = int(type_nr)
        else:
            self._type_nr = 0

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

    def morphed_zoom(self, image, zoom_size=DEFAULT_ZOOM, rotation=0,
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
    Annotations are represented as a list of Annotation objects"""

    def __init__( self, image_data=[], annotation_data=[] ):
        """Initialize.
            channel    = list or tuple of same size images
            annotation = list or tuple of Annotation objects"""
        self.channel = image_data
        self.annotation = annotation_data

    def __str__(self):
        return "AnnotatedImage (n_channels={:.0f}, n_annotations={:.0f}" \
                ")".format(self.n_channels, self.n_annotations)

    @property
    def n_channels(self):
        """Returns the (read-only) number of image channels"""
        return len(self.channel)

    @property
    def n_annotations(self):
        """Returns the (read-only) number of annotations"""
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
        if 'ROI' in mat_data.keys():
            nROIs = len(mat_data['ROI'][0])
            for c in range(nROIs):
                body = mat_data['ROI'][0][c]['body']
                body = np.array([body[:,1],body[:,0]]).transpose()
                type_nr = int(mat_data['ROI'][0][c]['type'][0][0])
                name = str(mat_data['ROI'][0][c]['typename'][0])
                group_nr = int(mat_data['ROI'][0][c]['group'][0][0])
                self.annotation.append( Annotation( body_pixels_yx=body,
                        annotation_name=name, type_nr=type_nr, group_nr=group_nr ) )
        elif 'ROIpy' in mat_data.keys():
            nROIs = len(mat_data['ROIpy'][0])
            for c in range(nROIs):
                body = mat_data['ROIpy'][0][c]['body'][0][0]
                body = np.array([body[:,1],body[:,0]]).transpose()
                type_nr = int(mat_data['ROIpy'][0][c]['type'][0][0][0][0])
                name = str(mat_data['ROIpy'][0][c]['typename'][0][0][0])
                group_nr = int(mat_data['ROIpy'][0][c]['group'][0][0][0][0])
                self.annotation.append( Annotation( body_pixels_yx=body,
                        annotation_name=name, type_nr=type_nr, group_nr=group_nr ) )

    def bodies(self, dilation_factor=0, mask_value=1):
        """Returns an image with annotation bodies masked"""
        annotated_bodies = np.zeros_like(self.channel[0])
        for nr in range(self.n_annotations):
            self.annotation[nr].mask_body(annotated_bodies,
                        dilation_factor=dilation_factor, mask_value=mask_value)
        return annotated_bodies

    def centroids(self, dilation_factor=0, mask_value=1):
        """Returns an image with annotation centroids masked"""
        annotated_centroids = np.zeros_like(self.channel[0])
        for nr in range(self.n_annotations):
            self.annotation[nr].mask_centroid(annotated_centroids,
                        dilation_factor=dilation_factor, mask_value=mask_value)
        return annotated_centroids

    def zoom(self, y, x, zoom_size=DEFAULT_ZOOM ):
        """Returns an image list, cropped to an area of tuple/list zoom_size
            around coordinates x and y
        zoom_size: (y size, x size), accepts only uneven numbers"""
        assert zoom_size[0] % 2 and zoom_size[1] % 2, \
            "zoom_size cannot contain even numbers: (%r,%r)" % zoom_size
        top_y  = np.int16( 1 + y - ((zoom_size[0]+1) / 2) )
        left_x = np.int16( 1 + x - ((zoom_size[1]+1) / 2) )
        ix_y = top_y + list(range( 0, zoom_size[0] ))
        ix_x = left_x + list(range( 0, zoom_size[1] ))
        zoom_list = []
        for ch in range(self.n_channels):
            zoom_list.append(self.channel[ch][ np.ix_(ix_y,ix_x) ])
        return zoom_list

    def zoom_1d(self, y, x, zoom_size=DEFAULT_ZOOM ):
        """Returns an single image vector, cropped to an area of tuple/list
            zoom_size around coordinates x & y, with all channels concatenated
        zoom_size: (y size, x size), accepts only uneven numbers"""
        zoom_list = self.zoom( y, x, zoom_size=zoom_size )
        return self.image_list_2d_to_1d( zoom_list )

    def image_list_2d_to_1d( self, image_list ):
        zoom_list_1d = []
        for ch in range(self.n_channels):
            zoom_list_1d.append(image_list[ch].ravel())
        return np.concatenate( zoom_list_1d )

    def image_list_1d_to_2d( self, lin_im, image_size=DEFAULT_ZOOM ):
        channels = np.split( lin_im, self.n_channels )
        image_list = []
        for ch in range(self.n_channels):
            image_list.append( np.reshape( channels[ch], image_size ) )
        return image_list

    def export_annotations_to_mat(self,file_name,file_path='.'):
        """Writes annotations to ROI_py.mat file"""
        roi_list = []
        for nr in range(self.n_annotations):
            roi_dict = {}
            roi_dict['nr']=nr
            roi_dict['group']=self.annotation[nr].group_nr
            roi_dict['type']=self.annotation[nr].type_nr
            roi_dict['typename']=self.annotation[nr].name
            roi_dict['x']=self.annotation[nr].x
            roi_dict['y']=self.annotation[nr].y
            roi_dict['size']=self.annotation[nr].size
            roi_dict['perimeter']=self.annotation[nr].perimeter
            body = np.array([self.annotation[nr].body[:,1],
                             self.annotation[nr].body[:,0]]).transpose()
            roi_dict['body']=body
            roi_list.append(roi_dict)
        savedata = {}
        savedata['ROIpy']=roi_list
        savemat(path.join(file_path,file_name),savedata)

    def load(self,file_name,file_path='.'):
        """Loads image and annotations from .npy file"""
        combined_annotated_image = np.load(path.join(file_path,file_name)).item()
        self.channel = combined_annotated_image['image_data']
        self.annotation = combined_annotated_image['annotation_data']

    def save(self,file_name,file_path='.'):
        """Saves image and annotations to .npy file"""
        combined_annotated_image = {}
        combined_annotated_image['image_data'] = self.channel
        combined_annotated_image['annotation_data'] = self.annotation
        np.save(path.join(file_path,file_name), combined_annotated_image)

    def centroid_detection_training_batch( self, m_samples=100,
                                           zoom_size=DEFAULT_ZOOM ):
        """Returns a 2d matrix (m samples x n pixels) with linearized data
            half of which is from within a centroid, and half from outside"""
        # Calculate number of positive and negative samples
        m_samples_pos = np.int16( m_samples * (0.5) )
        m_samples_neg = m_samples - m_samples_pos

        # Calculate size of image, and zoom
        (y_len,x_len) = self.channel[0].shape
        zoom_half_y = np.int16(zoom_size[0] / 2)
        zoom_half_x = np.int16(zoom_size[0] / 2)

        # Get coordinates of pixels within and outside of centroids
        (pix_x,pix_y) = np.meshgrid( np.arange(y_len),np.arange(x_len) )
        im_label = self.centroids( dilation_factor=1, mask_value=1)
        roi_positive_x = pix_x.ravel()[im_label.ravel() == 1]
        roi_positive_y = pix_y.ravel()[im_label.ravel() == 1]
        roi_negative_x = pix_x.ravel()[im_label.ravel() == 0]
        roi_negative_y = pix_y.ravel()[im_label.ravel() == 0]

        # Exclude all pixels that are within half-zoom from the border
        roi_positive_inclusion = np.logical_and( np.logical_and(
            roi_positive_x>zoom_half_x, roi_positive_x<(x_len-zoom_half_x) ),
            roi_positive_y>zoom_half_y, roi_positive_y<(y_len-zoom_half_y) )
        roi_positive_x = roi_positive_x[ roi_positive_inclusion ]
        roi_positive_y = roi_positive_y[ roi_positive_inclusion ]
        roi_negative_inclusion = np.logical_and( np.logical_and( np.logical_and(
            roi_negative_x>zoom_half_x, roi_negative_x<(x_len-(zoom_half_x)) ),
            roi_negative_y>zoom_half_y ), roi_negative_y<(y_len-(zoom_half_y)) )
        roi_negative_x = roi_negative_x[ roi_negative_inclusion ]
        roi_negative_y = roi_negative_y[ roi_negative_inclusion ]

        # Get list of random indices for pixel coordinates
        random_pos = np.random.choice( len(roi_positive_x),
                                        m_samples_pos, replace=False )
        random_neg = np.random.choice( len(roi_negative_x),
                                        m_samples_neg, replace=False )

        # Predefine output matrices
        samples = np.zeros( (m_samples,
            self.n_channels*zoom_size[0]*zoom_size[1]) )
        labels = np.zeros( (m_samples, 2) )
        count = 0

        # Positive examples
        for p in random_pos:
            samples[count,:] = self.zoom_1d(
                roi_positive_y[p], roi_positive_x[p], zoom_size )
            labels[count,1] = 1
            count = count + 1

        # Negative examples
        for p in random_neg:
            samples[count,:] = self.zoom_1d(
                roi_negative_y[p], roi_negative_x[p], zoom_size )
            labels[count,0] = 1
            count = count + 1

        return samples,labels


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
        return "AnnotatedImageSet (# Annotated Images = {:.0f}" \
                ")".format(self.n_annot_images)

    @property
    def n_annot_images(self):
        return len(self._an_im_list)

    @property
    def full_data_set(self):
        """Returns the (read-only) entire data set"""
        return self._an_im_list

    def load_data_dir(self,data_directory):
        """Loads all images and accompanying ROI.mat files from a
        single directory"""
        tiff_files = glob.glob(path.join(data_directory,'*.tiff'))
        mat_files = glob.glob(path.join(data_directory,'*.mat'))
        print("Loading .tiff and annotation files:")
        for f, (tiff_file, mat_file) in enumerate(zip(tiff_files,mat_files)):
            tiff_filepath, tiff_filename = path.split(tiff_file)
            mat_filepath, mat_filename = path.split(mat_file)
            print("{:2.0f}) {} -- {}".format(f+1,tiff_filename,mat_filename))
            anim = AnnotatedImage(image_data=[], annotation_data=[])
            anim.add_image_from_file(tiff_filename,tiff_filepath)
            anim.import_annotations_from_mat(mat_filename,mat_filepath)
            self._an_im_list.append(anim)
