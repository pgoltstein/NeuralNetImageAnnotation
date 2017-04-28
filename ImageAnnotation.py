#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 19:25:32 2016

Contains classes and functions that represent (sets of) images
and their annotations

Classes:
1. class Annotation(object):
    Class that holds an individual image annotation
2. class AnnotatedImage(object):
    Class that hold a multichannel image and its annotations
    Images are represented in a [x * y * n_channels] matrix
    Annotations are represented as a list of Annotation objects
3. class AnnotatedImageSet(object):
    Class that represents a dataset of annotated images and organizes
    the dataset for feeding in machine learning algorithms

Functions:

def zoom( image, y, x, zoom_size ):
    Crops an image to the area of tuple/list zoom_size around the
    supplied y, x coordinates. Pads out of range values.

def morph( image, rotation=0, scale_xy=(1,1), noise_level=0 ):
    Morphs image based on supplied parameters
    >> To do: recenter morphed image??

def morphed_zoom( image, y, x, zoom_size, pad_value=0,
                    rotation=0, scale_xy=(1,1), noise_level=0 ):
    Crops image or image list to area of zoom_size around centroid

def image2vec( image ):
    Concatenates a 2d image or image_list to a single 1d vector

def vec2image( lin_image, n_channels, image_size ):
    Constructs an image_list from a single 1d vector

def vec2RGB( lin_image, n_channels, image_size,
                    channel_order=(0,1,2), amplitude_scaling=(1,1,1) ):
    Constructs a 3d RGB image from a single 1d vector

def image_grid_RGB( lin_im_mat, n_x=10, n_y=6, image_size,
                    channel_order=(0,1,2),
                    amplitude_scaling=(1.33,1.33,1), line_color=0 ):
    Constructs a 3d numpy.ndarray tiled with a grid of RGB images. If
    more images are supplied that can be tiled, it chooses and displays
    a random subset.

@author: pgoltstein
"""



DEFAULT_ZOOM=(33,33)

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

import matplotlib.pyplot as plt
import seaborn as sns


########################################################################
### Functions
########################################################################

def zoom( image, y, x, zoom_size, normalize=False, pad_value=0 ):
    """Crops a(n) (list of) image(s) to the area of tuple/list zoom_size
    around the supplied y, x coordinates. Pads out of range values.
    image:      Single 2d numpy.ndarray or list of 2d numpy.ndarrays
    y, x:       Center coordinates
    zoom_size:  Size of zoomed image (y,x)
    normalize:  Normalizes to max
    pad_value:  Value for out of range coordinates
    returns zoomed image"""
    if isinstance(image,list):
        image_list = []
        for ch in range(len(image)):
            image_list.append( zoom( image[ch], y, x, zoom_size, pad_value ) )
        return image_list
    else:
        ix_y  = np.int16( np.round( 1 + y - ((zoom_size[0]+1) / 2) )
                    + np.arange( 0, zoom_size[0] ) )
        ix_x  = np.int16( np.round( 1 + x - ((zoom_size[1]+1) / 2) )
                    + np.arange( 0, zoom_size[0] ) )
        max_ix_exceed = -1 * np.min( ( np.min(ix_y), np.min(ix_x),
            image.shape[0]-np.max(ix_y)-1, image.shape[1]-np.max(ix_x)-1 ) )
        if max_ix_exceed > 0:
            image_temp = np.zeros((image.shape+max_ix_exceed+1))+pad_value
            image_temp[0:image.shape[0],0:image.shape[1]] = image
            if normalize:
                zoom_im = image_temp[ np.ix_(ix_y,ix_x) ]
                return zoom_im / zoom_im.max()
            else:
                return image_temp[ np.ix_(ix_y,ix_x) ]
        else:
            if normalize:
                zoom_im = image[ np.ix_(ix_y,ix_x) ]
                return zoom_im / zoom_im.max()
            else:
                return image[ np.ix_(ix_y,ix_x) ]

def morph( image, rotation=0, scale_xy=(1,1), noise_level=0 ):
    """Morphs (list of) image(s) based on supplied parameters
    image:        Single 2d numpy.ndarray or list of 2d numpy.ndarrays
    rotation:     Rotation of annotation in degrees (0-360 degrees)
    scale_xy:     Determines fractional scaling on x/y axis.
                    Min-Max = (0.5,0.5) - (2,2)
    noise_level:  Standard deviation of random Gaussian noise
    returns morped_image"""
    if isinstance( image, list ):
        image_list = []
        for ch in range(len(image)):
            image_list.append( morph( image[ch],
                                        rotation, scale_xy, noise_level ) )
        return image_list
    else:
        # Rotate
        if rotation != 0:
            image = ndimage.interpolation.rotate(image, rotation, reshape=False)
        # Scale
        if scale_xy[0] != 1 or scale_xy[1] != 1:
            image = ndimage.interpolation.zoom( image, scale_xy )
        # Add noise
        if noise_level:
            noise_mask = np.random.normal(size=image.shape) * noise_level
            image = image + (image * noise_mask)
        return image

def morphed_zoom( image, y, x, zoom_size, pad_value=0, normalize=False,
                    rotation=0, scale_xy=(1,1), noise_level=0 ):
    """Crops image or image list to area of zoom_size around centroid
    image:        Single 2d numpy.ndarray or list of 2d numpy.ndarrays
    y, x:         Center coordinates
    zoom_size:    (y size, x size)
    pad_value:    Value for out of range coordinates
    normalize:    Normalizes to max
    rotation:     Rotation of annotation in degrees (0-360 degrees)
    scale_xy:     Determines fractional scaling on x/y axis.
                  Min-Max = (0.5,0.5) - (2,2)
    noise_level:  Level of random noise
    returns tuple holding (morped_zoom, morped_annotation)"""
    im = zoom( image=image, y=y, x=x,
        zoom_size=(zoom_size[0]*3,zoom_size[1]*3),
        normalize=False, pad_value=pad_value )
    im = morph( image=im,
            rotation=rotation, scale_xy=scale_xy, noise_level=noise_level )
    if isinstance( im, list ):
        y_pos, x_pos = (im[0].shape[0]-1)/2, (im[0].shape[1]-1)/2
    else:
        y_pos, x_pos = (im.shape[0]-1)/2, (im.shape[1]-1)/2
    return zoom( im, y=y_pos, x=x_pos, zoom_size=zoom_size,
                        normalize=normalize, pad_value=pad_value )

def image2vec( image ):
    """Concatenates a 2d image or image_list to a single 1d vector
    image:  single 2d numpy.ndarray or list of 2d numpy.ndarrays
    returns 1d vector with all pixels concatenated"""
    image_1d = []
    if isinstance( image, list ):
        for ch in range(len(image)):
            image_1d.append(image[ch].ravel())
    else:
            image_1d.append(image.ravel())
    return np.concatenate( image_1d )

def vec2image( lin_image, n_channels, image_size ):
    """Constructs an image_list from a single 1d vector
    lin_image:   1d image vector with all pixels concatenated
    n_channels:  Number of image channels
    image_size:  2 dimensional size of the image (y,x)
    returns single or list of 2d numpy.ndarrays"""
    if n_channels > 1:
        channels = np.split( lin_image, n_channels )
        image = []
        for ch in range(n_channels):
            image.append( np.reshape( channels[ch], image_size ) )
    else:
        image = np.reshape( lin_image, image_size )
    return image

def vec2RGB( lin_image, n_channels, image_size,
                    channel_order=(0,1,2), amplitude_scaling=(1,1,1) ):
    """Constructs a 3d RGB image from a single 1d vector
    lin_image:   1d image vector with all pixels concatenated
    n_channels:  Number of image channels
    image_size:  2 dimensional size of the image (y,x)
    channel_order:  tuple indicating which channels are R, G and B
    amplitude_scaling:  Additional scaling of each channel separately
    returns 3d numpy.ndarray"""
    image = vec2image( lin_image, n_channels, image_size )
    RGB = np.zeros((image_size[0],image_size[1],3))
    if n_channels > 1:
        for nr,ch in enumerate(channel_order):
            RGB[:,:,nr] = image[ch]
    else:
        for ch in range(3):
            RGB[:,:,ch] = image
    return RGB

def image_grid_RGB( lin_images, n_channels, image_size, annotation_nrs=None,
                    n_x=10, n_y=6, channel_order=(0,1,2), auto_scale=False,
                    amplitude_scaling=(1.33,1.33,1), line_color=0 ):
    """ Constructs a 3d numpy.ndarray tiled with a grid of RGB images. If
        more images are supplied that can be tiled, it chooses and displays
        a random subset.
    lin_images:         2d matrix with on each row an image vector with all
                        pixels concatenated or a list with images
    n_channels:  Number of image channels
    image_size:         2 dimensional size of the image (y,x)
    annotation_nrs:     List with nr of the to be displayed annotations
    n_x:                Number of images to show on x axis of grid
    n_y:                Number of images to show on y axis of grid
    channel_order:      Tuple indicating which channels are R, G and B
    auto_scale:        Scale each individual image to its maximum (T/F)
    amplitude_scaling:  Intensity scaling of each color channel
    line_color:         Intensity (gray scale) of line between images
    Returns numpy.ndarray (x,y,RGB)
    """

    # Get indices of images to show
    if annotation_nrs is None:
        annotation_nrs = list(range(lin_images.shape[0]))
    n_images = len(annotation_nrs)
    if n_images <= n_x*n_y:
        im_ix = list(range(n_images))
    else:
        im_ix = np.random.choice( n_images, n_x*n_y, replace=False )

    # Get coordinates of where images will go
    y_coords = []
    offset = 0
    for i in range(n_y):
        offset = i * (image_size[0] + 1)
        y_coords.append(offset+np.array(range(image_size[0])))
    max_y = np.max(y_coords[i]) + 1

    x_coords = []
    offset = 0
    for i in range(n_x):
        offset = i * (image_size[1] + 1)
        x_coords.append(offset+np.array(range(image_size[1])))
    max_x = np.max(x_coords[i]) + 1

    rgb_coords = np.array(list(range(3)))

    # Fill grid
    im_count = 0
    center_shift = []
    grid = np.zeros((max_y,max_x,3))+line_color
    for y in range(n_y):
        for x in range(n_x):
            if im_count < n_images:
                rgb_im = vec2RGB( lin_images[ im_ix[ im_count ], : ],
                    n_channels=n_channels, image_size=image_size,
                    channel_order=channel_order,
                    amplitude_scaling=amplitude_scaling )
                if auto_scale:
                    rgb_im = rgb_im / rgb_im.max()
                grid[np.ix_(y_coords[y],x_coords[x],rgb_coords)] = rgb_im
                center_shift.append( \
                    ( y_coords[y][0] + (0.5*image_size[0]) -0.5,
                      x_coords[x][0] + (0.5*image_size[0]) -0.5 ) )
            else:
                break
            im_count += 1
    return grid, center_shift

def split_samples( m_samples, n_groups, ratios=None ):
    """Splits the total number of samples into n_groups according to the
    relative ratios (compensates for rounding errors)
    m_samples:  Total number of samples
    n_groups:   Number of sample groups to return
    ratios:     List with relative ratio of each group
    returns list with sample counts per group"""

    if ratios is None:
        ratios = n_groups * [ (1/n_groups),]
    else:
        ratios = np.array(ratios)
        ratios = ratios/ratios.sum()

    # Calculate minimum number of positive and negative samples and round err
    g_samples = []
    g_round_ratios = []
    for g in range(n_groups):
        g_samples.append( np.int16( m_samples * ratios[g] ) )
        g_round_ratios.append( (m_samples * ratios[g]) % 1 )

    # Find how many samples are still missing
    n_missing = m_samples - np.sum(g_samples)

    # Assign missing samples by relative remainder fractional chance to groups
    if n_missing > 0:
        ratio_group_ids = list(range(len(g_round_ratios)))
        for s in range(n_missing):
            rand_num = np.random.rand(1)
            for g in range(len(g_round_ratios)):
                if rand_num < np.sum(g_round_ratios[:(g+1)]):
                    g_samples[ratio_group_ids[g]] += 1
                    del g_round_ratios[g]
                    del ratio_group_ids[g]
                    break
    return g_samples


def get_labeled_pixel_coordinates( bin_image, exclude_border=(0,0,0,0) ):
    """Get the x and y pixels coordinates of all labeled pixels in a
        binary image, excluding the pixels outside of the border
    bin_image:         Binary image (numpy array)
    exclude_border:    exclude annotations that are a certain distance
                       to each border. Pix from (left, right, up, down)
    returns tuple y_pix,x_pix with numpy.array pixel coordinates"""

    # Get lists with all pixel coordinates
    y_res,x_res = bin_image.shape
    (pix_x,pix_y) = np.meshgrid(np.arange(y_res),np.arange(x_res))

    # Get lists with coordinates of all labeled pixels
    lab_pix_x = pix_x.ravel()[bin_image.ravel() == 1]
    lab_pix_y = pix_y.ravel()[bin_image.ravel() == 1]

    # Exclude all pixels that are too close to the border
    if np.max(exclude_border) > 0:
        include_pix = \
            np.logical_and( np.logical_and( np.logical_and(
                lab_pix_x > exclude_border[0],
                lab_pix_x < (x_res-exclude_border[1]) ),
                lab_pix_y > exclude_border[2] ),
                lab_pix_y < (y_res-exclude_border[3]) )
        lab_pix_x = lab_pix_x[ include_pix ]
        lab_pix_y = lab_pix_y[ include_pix ]

    # Return pixel coordinates
    return lab_pix_y,lab_pix_x


########################################################################
### Class Annotation
########################################################################

class Annotation(object):
    """Class that holds an individual image annotation"""

    def __init__( self, body_pixels_yx, annotation_name="Neuron",
                                    type_nr=1, group_nr=0):
        """Initialize.
            body_pixels_yx: list/tuple of (y,x) coordinates
                            or a 2d binary image mask
            annotation_name: string
            type_nr: int
            group_nr: int
        """
        # Store supplied parameters
        if isinstance( body_pixels_yx, list ):
            self.body = np.array(np.int16(body_pixels_yx))
        elif body_pixels_yx.shape[1] == 2:
            self.body = np.array(np.int16(body_pixels_yx))
        else:
            self.body = np.transpose( \
                np.nonzero( np.array( np.int16(body_pixels_yx) ) ) )
        self.name = str(annotation_name)
        self.type_nr = int(type_nr)
        self.group_nr = int(group_nr)

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
        temp_mask = np.zeros( self._body.max(axis=0)+3 )
        temp_mask[ self._body[:,0]+1, self._body[:,1]+1 ] = 1
        self._perimeter = measure.find_contours(temp_mask, 0.5)[0]-1
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
        """Sets group number"""
        self._group_nr = int(group_nr)

    @property
    def type_nr(self):
        """Returns read-only type number"""
        return self._type_nr

    @type_nr.setter
    def type_nr(self,type_nr):
        """Sets type number to integer"""
        self._type_nr = int(type_nr)

    @property
    def perimeter(self):
        """Returns read-only stored list of perimeter (y,x) coordinates"""
        return self._perimeter

    @property
    def size(self):
        """Returns read-only size of annotation (number of pixels)"""
        return self._size

    def zoom(self, image, zoom_size, pad_value=0, normalize=False ):
        """Crops image to area of tuple/list zoom_size around centroid
        image:      Single 2d numpy.ndarray
        zoom_size:  (y size, x size)
        pad_value:  Value for out of range coordinates
        normalize:  Normalizes to max
        returns zoomed image"""
        return zoom( image=image, y=self._y, x=self._x, zoom_size=zoom_size,
                        normalize=normalize, pad_value=pad_value )

    def morphed_zoom(self, image, zoom_size, pad_value=0, normalize=False,
                        rotation=0, scale_xy=(1,1), noise_level=0 ):
        """Crops image to area of tuple/list zoom_size around centroid
        image:        Single 2d numpy.ndarray
        zoom_size:    (y size, x size)
        pad_value:    Value for out of range coordinates
        normalize:    Normalizes to max
        rotation:     Rotation of annotation in degrees (0-360 degrees)
        scale_xy:     Determines fractional scaling on x/y axis.
                      Min-Max = (0.5,0.5) - (2,2)
        noise_level:  Level of random noise
        returns tuple holding (morped_zoom, morped_annotation)"""
        return morphed_zoom( image, self._y, self._x, zoom_size=zoom_size,
                    pad_value=pad_value, normalize=normalize, rotation=rotation,
                    scale_xy=scale_xy, noise_level=noise_level )

    def mask_body(self, image, dilation_factor=0,
                        mask_value=1, keep_centroid=True):
        """Draws mask of all body pixels in image
        image:            Single 2d numpy.ndarray
        dilation_factor:  >0 for dilation, <0 for erosion
        mask_value:       Value to place in image
        keep_centroid:    Prevents mask from disappearing altogether with
                          negative dilation factors
        returns masked image"""
        if dilation_factor==0:
            # Just mask the incoming image
            image[ self._body[:,0], self._body[:,1] ] = mask_value
        else:
            # Draw mask on temp image, dilate, get pixels, then draw in image
            temp_mask = np.zeros_like(image,dtype=bool)
            temp_mask[ self._body[:,0],self._body[:,1] ] = True
            if dilation_factor>0:
                for _ in range(dilation_factor):
                    temp_mask = ndimage.binary_dilation(temp_mask)
            elif dilation_factor<0:
                for _ in range(-1*dilation_factor):
                    temp_mask = ndimage.binary_erosion(temp_mask)
            temp_body = np.array(np.where(temp_mask == True)).transpose()
            image[ temp_body[:,0], temp_body[:,1] ] = mask_value
            if keep_centroid:
                image[self._y.astype(int),self._x.astype(int)] = mask_value

    def mask_centroid(self, image, dilation_factor=0, mask_value=1):
        """Draws mask of centroid pixel in image
        image:            Single 2d numpy.ndarray
        dilation_factor:  >0 for padding the centroid with surrounding points
        mask_value:       Value to place in image
        returns masked image"""
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


########################################################################
### Class AnnotatedImage
########################################################################

class AnnotatedImage(object):
    """Class that hold a multichannel image and its annotations
    Images are represented in a list of [x * y] matrices
    Annotations are represented as a list of Annotation objects"""

    def __init__( self, image_data=None, annotation_data=None,
        exclude_border=None, detected_centroids=None, detected_bodies=None,
        labeled_centroids=None, labeled_bodies=None,
        include_annotation_typenr=None, downsample=None):
        """Initialize image list and channel list
            channel:             List or tuple of same size images
            annotation:          List or tuple of Annotation objects
            exclude_border:      4-Tuple containing border exclusion region
                                 (left,right,top,bottom), dictionary, or
                                 file name of mat file holding the parameters
                                 as separate variables
            detected_centroids:  Binary image with centroids labeled
            detected_bodies:     Binary image with bodies labeled
            labeled_centroids:   Image with annotation centroids labeled by number
            labeled_bodies:      Image with annotation bodies labeled by number
            downsample:          Downsample to be imported images, borders
                                 and ROI's by a certain factor
            """
        self._downsample = downsample
        self._bodies = None
        self._body_dilation_factor = 0
        self._centroids = None
        self._centroid_dilation_factor = 0
        self._include_annotation_typenrs = None
        self._y_res = 0
        self._x_res = 0
        self._channel = []
        self._annotation = []
        self._exclude_border = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        self._exclude_border_tuple = (0,0,0,0)
        if image_data is not None:
            self.channel = image_data
        if annotation_data is not None:
            self.annotation = annotation_data
        if exclude_border is not None:
            self.exclude_border = exclude_border
        self.detected_centroids = detected_centroids
        self.detected_bodies = detected_bodies
        self.labeled_centroids = labeled_centroids
        self.labeled_bodies = labeled_bodies

    def __str__(self):
        return "AnnotatedImage (#ch={:.0f}, #ann={:.0f}, " \
                "brdr={:d},{:d},{:d},{:d})".format( self.n_channels,
                self.n_annotations, self.exclude_border['left'],
                self.exclude_border['right'], self.exclude_border['top'],
                self.exclude_border['bottom'])

    # **********************************
    # *****  Describing properties *****
    @property
    def y_res(self):
        """Returns the (read-only) size of the y-dimension of the images"""
        return self._y_res

    @property
    def x_res(self):
        """Returns the (read-only) size of the x-dimension of the images"""
        return self._x_res

    @property
    def im_size(self):
        """Returns the (read-only) size of the image as tuple"""
        return (self._y_res,self._x_res)

    @property
    def n_channels(self):
        """Returns the (read-only) number of image channels"""
        return len(self._channel)

    @property
    def n_annotations(self):
        """Returns the (read-only) number of annotations"""
        return len(self._annotation)

    @property
    def downsamplingfactor(self):
        """Returns the (read-only) downsampling factor"""
        return self._downsample

    @property
    def class_labels(self):
        """Returns the class labels that are set for training"""
        class_labels = [0,]
        class_labels.extend(list(self.include_annotation_typenrs))
        return class_labels

    # ************************************
    # *****  Handling the image data *****
    @property
    def channel(self):
        """Returns list with all image channels"""
        return self._channel

    @channel.setter
    def channel(self, image_data):
        """Sets the internal list with all image channels to np.ndarray copies
        of the supplied list with -to numpy.ndarray convertable- image data
        image_data:  single image, or list with images that are converable to
        a numpy.ndarray"""
        self._channel = []
        self._bodies = None
        self._centroids = None
        y_res_old,x_res_old = self.y_res,self.x_res
        if isinstance( image_data, list):
            for im in image_data:
                if self.downsamplingfactor is not None:
                    self._channel.append( ndimage.interpolation.zoom( \
                        np.array(im), 1/self.downsamplingfactor ) )
                else:
                    self._channel.append( np.array(im) )
        else:
            if self.downsamplingfactor is not None:
                self._channel.append( ndimage.interpolation.zoom( \
                    np.array(image_data), 1/self.downsamplingfactor ) )
            else:
                self._channel.append( np.array(image_data) )
        self._y_res,self._x_res = self._channel[0].shape

        # Update masks if there are annotations and the image resolution changed
        if self.n_annotations > 0 and ( (y_res_old != self.y_res)
                                    or (x_res_old != self.x_res) ):
            self._set_bodies()
            self._set_centroids()

    @property
    def exclude_border(self):
        """Returns dictionary with border exclusion parameters"""
        return self._exclude_border

    @property
    def exclude_border_tuple(self):
        """Returns dictionary with border exclusion parameters"""
        return self._exclude_border_tuple

    @exclude_border.setter
    def exclude_border( self, exclude_border ):
        """Sets the exclude_border parameter dictionary
        exclude_border: 4-Tuple containing border exclusion region (left,
                        right,top,bottom), dictionary, or file name of mat
                        file holding the parameters as separate variables
                        named ExclLeft, ExclRight, ExclTop, ExclBottom
        Returns dictionary {'left': #, 'right': #, 'top': #, 'bottom': #}
        """
        if isinstance(exclude_border,list) or isinstance(exclude_border,tuple):
            self._exclude_border['left'] = exclude_border[0]
            self._exclude_border['right'] = exclude_border[1]
            self._exclude_border['top'] = exclude_border[2]
            self._exclude_border['bottom'] = exclude_border[3]
        elif isinstance(exclude_border,dict):
            self._exclude_border['left'] = exclude_border['left']
            self._exclude_border['right'] = exclude_border['right']
            self._exclude_border['top'] = exclude_border['top']
            self._exclude_border['bottom'] = exclude_border['bottom']
        elif isinstance(exclude_border,str):
            mat_data = loadmat(exclude_border)
            self._exclude_border['left'] = int(mat_data['ExclLeft'])
            self._exclude_border['right'] = int(mat_data['ExclRight'])
            self._exclude_border['top'] = int(mat_data['ExclTop'])
            self._exclude_border['bottom'] = int(mat_data['ExclBottom'])
        if self.downsamplingfactor is not None:
            self._exclude_border['left'] = \
                int(np.round(self._exclude_border['left']/self.downsamplingfactor))
            self._exclude_border['right'] = \
                int(np.round(self._exclude_border['right']/self.downsamplingfactor))
            self._exclude_border['top'] = \
                int(np.round(self._exclude_border['top']/self.downsamplingfactor))
            self._exclude_border['bottom'] = \
                int(np.round(self._exclude_border['bottom']/self.downsamplingfactor))
        self._exclude_border_tuple = \
            ( int(self._exclude_border['left']), int(self._exclude_border['right']),
              int(self._exclude_border['top']), int(self._exclude_border['bottom']) )

    def add_image_from_file(self, file_name, file_path='.',
                                normalize=True, use_channels=None):
        """Loads image or matlab cell array, scales individual channels to
        max (1), and adds it as a new image channel
        file_name:  String holding name of image file
        file_path:  String holding file path
        normalize:  Normalize to maximum of image
        use_channels:  tuple holding channel numbers/order to load (None=all)
        """
        y_res_old,x_res_old = self.y_res,self.x_res

        # Load from .mat file with cell array
        if str(file_name[-4:]) == ".mat":
            mat_data = loadmat(path.join(file_path,file_name))
            n_channels = mat_data['Images'].shape[1]
            if use_channels is None:
                use_channels = list(range(n_channels))
            for ch in use_channels:
                im_x = np.float64(np.array(mat_data['Images'][0,ch]))
                if normalize:
                    im_x = im_x / im_x.max()
                if self.downsamplingfactor is not None:
                    self._channel.append( ndimage.interpolation.zoom( \
                        im_x, 1/self.downsamplingfactor ) )
                else:
                    self._channel.append(im_x)

        # Load from actual image
        else:
            im = np.float64(imread(path.join(file_path,file_name)))

            # Perform normalization (max=1) and add to channels
            if im.ndim == 3:
                n_channels = np.size(im,axis=2)
                if use_channels is None:
                    use_channels = list(range(n_channels))
                for ch in use_channels:
                    im_x = im[:,:,ch]
                    if normalize:
                        im_x = im_x / im_x.max()
                    if self.downsamplingfactor is not None:
                        self._channel.append( ndimage.interpolation.zoom( \
                            im_x, 1/self.downsamplingfactor ) )
                    else:
                        self._channel.append(im_x)
            else:
                if normalize:
                    im = im / im.max()
                if self.downsamplingfactor is not None:
                    self._channel.append( ndimage.interpolation.zoom( \
                        im, 1/self.downsamplingfactor ) )
                else:
                    self._channel.append(im)

        # Set resolution
        self._y_res,self._x_res = self._channel[0].shape

        # Update masks if there are annotations and the image resolution changed
        if self.n_annotations > 0 and ( (y_res_old != self.y_res)
                                    or (x_res_old != self.x_res) ):
            self._set_bodies()
            self._set_centroids()

    def RGB( self, channel_order=(0,1,2), amplitude_scaling=(1,1,1) ):
        """Constructs an RGB image from the image list
        channel_order:      tuple indicating which channels are R, G and B
        amplitude_scaling:  Additional scaling of each channel separately
        returns 3d numpy.ndarray"""
        RGB = np.zeros((self.y_res,self.x_res,3))
        for ch in range(3):
            if channel_order[ch] < self.n_channels:
                RGB[:,:,ch] = self.channel[channel_order[ch]]
        return RGB

    def crop( self, left, top, width, height ):
        """Crops the image channels, annotations and borders
        left:   Left most pixel in cropped image (0 based)
        top:    Top most pixel in cropped image (0 based)
        width:  Width of cropped region
        height: Height of cropped region
        """

        # Crop channels
        new_channel_list = []
        for nr in range(self.n_channels):
            new_channel_list.append( self._channel[nr][top:top+height,left:left+width] )

        # Crop annotations
        new_annotation_list = []
        for an in self.annotation:
            an_mask = np.zeros((self.y_res,self.x_res))
            an.mask_body( image=an_mask )
            new_an_mask = an_mask[top:top+height,left:left+width]
            if new_an_mask.sum() > 0:
                new_annotation_list.append( Annotation( body_pixels_yx=new_an_mask,
                    annotation_name=an.name, type_nr=an.type_nr, group_nr=an.group_nr) )

        # Crop borders
        brdr = self.exclude_border.copy()
        brdr['left'] = np.max( [ brdr['left']-left, 0 ] )
        brdr['top'] = np.max( [ brdr['top']-top, 0 ] )
        crop_from_right = self.x_res-(left+width)
        brdr['right'] = np.max( [ brdr['right']-crop_from_right, 0 ] )
        crop_from_bottom = self.x_res-(left+width)
        brdr['bottom'] = np.max( [ brdr['bottom']-crop_from_bottom, 0 ] )

        # Update annotations and channels
        self.annotation = new_annotation_list
        self.channel = new_channel_list
        self.exclude_border = brdr


    # *****************************************
    # *****  Handling the annotation data *****
    @property
    def annotation(self):
        """Returns list with all image annotations"""
        return self._annotation

    @annotation.setter
    def annotation(self, annotation_data):
        """Sets the internal list of all image annotations to copies of the
        supplied list of class annotation() annotations
        annotation_data:  instance of, or list with annotations of the
                          annotation class"""
        self._annotation = []
        if not isinstance( annotation_data, list):
            annotation_data = [annotation_data]
        type_nr_list = []
        for an in annotation_data:
            if self.downsamplingfactor is not None:
                body_pixels = np.round(an.body / self.downsamplingfactor)
            else:
                body_pixels = an.body
            self._annotation.append( Annotation(
                body_pixels_yx=body_pixels,
                annotation_name=an.name,
                type_nr=an.type_nr,
                group_nr=an.group_nr) )
            type_nr_list.append(an.type_nr)
        # Update masks if there is at least one image channel
        if self.include_annotation_typenrs is None:
            self.include_annotation_typenrs = type_nr_list
        if self.n_channels > 0:
            self._set_bodies()
            self._set_centroids()

    def import_annotations_from_mat(self, file_name, file_path='.'):
        """Reads data from ROI.mat file and fills the annotation_list.
        file_name:     String holding name of ROI file
        file_path:     String holding file path
        """

        # Load mat file with ROI data
        mat_data = loadmat(path.join(file_path,file_name))
        annotation_list = []
        type_nr_list = []
        nROIs = len(mat_data['ROI'][0])
        for c in range(nROIs):
            body = mat_data['ROI'][0][c]['body']
            body = np.array([body[:,1],body[:,0]]).transpose()
            body = body-1 # Matlab (1-index) to Python (0-index)
            type_nr = int(mat_data['ROI'][0][c]['type'][0][0])
            name = str(mat_data['ROI'][0][c]['typename'][0])
            group_nr = int(mat_data['ROI'][0][c]['group'][0][0])
            annotation_list.append( Annotation( body_pixels_yx=body,
                    annotation_name=name, type_nr=type_nr, group_nr=group_nr ) )
            type_nr_list.append(type_nr)
        if self.include_annotation_typenrs is None:
            self.include_annotation_typenrs = type_nr_list
        self.annotation = annotation_list

    def export_annotations_to_mat(self, file_name,
                                        file_path='.', upsample=None):
        """Writes annotations to ROI.mat file
        file_name:  String holding name of ROI file
        file_path:  String holding file path
        upsample:   Upsampling factor"""

        if upsample is not None:
            upsamplingfactor = upsample
        elif self.downsamplingfactor is not None:
            print("AnnotatedImage was downsampled by factor of {}".format( \
                self.downsamplingfactor) + ", upsampling ROI's for export ")
            upsamplingfactor = self.downsamplingfactor
        else:
            upsamplingfactor = None

        # Upsample ROI's before export
        if upsamplingfactor is not None:
            annotation_export_list = []
            for an in self.annotation:
                annotation_mask = np.zeros_like(self._channel[0])
                an.mask_body(image=annotation_mask)
                annotation_mask = ndimage.interpolation.zoom( \
                    annotation_mask, self.downsamplingfactor )
                annotation_export_list.append( Annotation(
                    body_pixels_yx=annotation_mask>0.5, annotation_name=an.name,
                    type_nr=an.type_nr, group_nr=an.group_nr) )
        else:
            annotation_export_list = self.annotation

        # Export ROIs
        nrs = []
        groups = []
        types = []
        typenames = []
        xs = []
        ys = []
        sizes = []
        perimeters = []
        bodys = []
        for nr,an in enumerate(annotation_export_list):
            nrs.append(nr)
            groups.append(an.group_nr)
            types.append(an.type_nr)
            typenames.append(an.name)
            xs.append(an.x+1)
            ys.append(an.y+1)
            sizes.append(an.size)
            perimeter = np.array( \
                [an.perimeter[:,1],an.perimeter[:,0]] ).transpose()+1
            perimeters.append(perimeter)
            body = np.array( [an.body[:,1],an.body[:,0]] ).transpose()+1
            bodys.append(body)
        savedata = np.core.records.fromarrays( [ nrs, groups, types,
            typenames, xs, ys, sizes, perimeters, bodys ],
            names = [ 'nr', 'group', 'type', 'typename', 'x', 'y',
                        'size', 'perimeter', 'body'] )
        savemat(path.join(file_path,file_name), {'ROI': savedata} )
        print("Exported annotations to: {}".format(
                                    path.join(file_path,file_name)+".mat"))

    # ******************************************
    # *****  Handling the annotated bodies *****
    @property
    def bodies(self):
        """Returns an image with annotation bodies masked"""
        return self._bodies

    @property
    def bodies_typenr(self):
        """Returns an image with annotation bodies masked by type_nr"""
        return self._bodies_type_nr

    def _set_bodies(self):
        """Sets the internal body annotation mask with specified parameters"""
        self._bodies = np.zeros_like(self._channel[0])
        self._bodies_type_nr = np.zeros_like(self._channel[0])
        for nr in range(self.n_annotations):
            if self._annotation[nr].type_nr in self.include_annotation_typenrs:
                self._annotation[nr].mask_body(self._bodies,
                    dilation_factor=self._body_dilation_factor,
                    mask_value=nr+1, keep_centroid=True)
                self._bodies_type_nr[self._bodies==nr+1] = \
                                        self._annotation[nr].type_nr

    @property
    def body_dilation_factor(self):
        """Returns the body dilation factor"""
        return(self._body_dilation_factor)

    @body_dilation_factor.setter
    def body_dilation_factor(self, dilation_factor):
        """Updates the internal body annotation mask with dilation_factor"""
        self._body_dilation_factor = dilation_factor
        self._set_bodies()


    # *********************************************
    # *****  Handling the annotated centroids *****
    @property
    def centroids(self):
        """Returns an image with annotation centroids masked"""
        return self._centroids

    @property
    def centroids_typenr(self):
        """Returns an image with annotation centroids masked by type_nr"""
        return self._centroids_type_nr

    def _set_centroids(self):
        """Sets the internal centroids annotation mask with specified
        parameters"""
        self._centroids = np.zeros_like(self._channel[0])
        self._centroids_type_nr = np.zeros_like(self._channel[0])
        for nr in range(self.n_annotations):
            if self._annotation[nr].type_nr in self.include_annotation_typenrs:
                self._annotation[nr].mask_centroid(self._centroids,
                    dilation_factor=self._centroid_dilation_factor,
                    mask_value=nr+1)
                self._centroids_type_nr[self._centroids==nr+1] = \
                                            self._annotation[nr].type_nr

    @property
    def centroid_dilation_factor(self):
        """Returns the centroid dilation factor"""
        return(self._centroid_dilation_factor)

    @centroid_dilation_factor.setter
    def centroid_dilation_factor(self, dilation_factor):
        """Updates the internal centroid annotation mask with dilation_factor"""
        self._centroid_dilation_factor = dilation_factor
        self._set_centroids()


    # ***************************************************
    # *****  Loading and saving of Annotated Images *****
    def load(self,file_name,file_path='.'):
        """Loads image and annotations from .npy file"""
        combined_annotated_image = np.load(path.join(file_path,file_name)).item()
        self.channel = combined_annotated_image['image_data']
        self.annotation = combined_annotated_image['annotation_data']
        self.exclude_border = combined_annotated_image['exclude_border']
        self.include_annotation_typenrs = \
                        combined_annotated_image['include_annotation_typenrs']
        self.detected_centroids = combined_annotated_image['detected_centroids']
        self.detected_bodies = combined_annotated_image['detected_bodies']
        self.labeled_centroids = combined_annotated_image['labeled_centroids']
        self.labeled_bodies = combined_annotated_image['labeled_bodies']
        print("Loaded AnnotatedImage from: {}".format(
                                    path.join(file_path,file_name)))

    def save(self,file_name,file_path='.'):
        """Saves image and annotations to .npy file"""
        combined_annotated_image = {}
        combined_annotated_image['image_data'] = self.channel
        combined_annotated_image['annotation_data'] = self.annotation
        combined_annotated_image['exclude_border'] = self.exclude_border
        combined_annotated_image['include_annotation_typenrs'] = \
                                            self.include_annotation_typenrs
        combined_annotated_image['detected_centroids'] = self.detected_centroids
        combined_annotated_image['detected_bodies'] = self.detected_bodies
        combined_annotated_image['labeled_centroids'] = self.labeled_centroids
        combined_annotated_image['labeled_bodies'] = self.labeled_bodies
        np.save(path.join(file_path,file_name), combined_annotated_image)
        print("Saved AnnotatedImage as: {}".format(
                                    path.join(file_path,file_name)+".npy"))


    # ************************************************
    # *****  Generate NN training/test data sets *****

    @property
    def include_annotation_typenrs(self):
        """Includes only ROI's with certain typenrs in body and centroid masks
        """
        return self._include_annotation_typenrs

    @include_annotation_typenrs.setter
    def include_annotation_typenrs(self, include_typenrs):
        """Sets the nrs to include, removes redundancy by using sets"""

        if isinstance(include_typenrs,int):
            annotation_typenrs = set([include_typenrs,])
        elif include_typenrs is None:
            type_nr_list = []
            for an in self.annotation:
                type_nr_list.append(an.type_nr)
            annotation_typenrs = set(type_nr_list)
        else:
            annotation_typenrs = set(include_typenrs)

        if 0 in annotation_typenrs:
            annotation_typenrs.remove(0)

        self._include_annotation_typenrs = annotation_typenrs
        if self.n_channels > 0:
            self._set_centroids()
            self._set_bodies()

    def get_batch( self, zoom_size, annotation_type='Bodies',
            m_samples=100, return_annotations=False,
            sample_ratio=None, annotation_border_ratio=None,
            normalize_samples=False, segment_all=False,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None ):
        """Constructs a 2d matrix (m samples x n pixels) with linearized data
            half of which is from within an annotation, and half from outside
            zoom_size:         2 dimensional size of the image (y,x)
            annotation_type:   'Bodies' or 'Centroids'
            m_samples:         number of training samples
            return_annotations:  Returns annotations in addition to
                                 samples and labels. If False, returns empty
                                 list. Otherwise set to 'Bodies' or 'Centroids'
            sample_ratio:        List with ratio of samples per groups (sum=1)
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            normalize_samples: Scale each individual channel to its maximum
            segment_all:       Segments all instead of single annotations (T/F)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            Returns tuple with samples as 2d numpy matrix, labels as
            2d numpy matrix and if requested annotations as 2d numpy matrix
            or otherwise an empty list as third item"""

        # Calculate number of samples per class
        class_labels = sorted(self.class_labels)
        n_classes = len(class_labels)
        if sample_ratio is not None:
            if len(sample_ratio) > n_classes:
                sample_ratio = sample_ratio[:n_classes]
        m_class_samples = split_samples(
            m_samples, n_classes, ratios=sample_ratio )

        # Get number of border annotations (same strategy as above)
        if annotation_border_ratio is not None:
            m_class_borders = list(range(n_classes))
            for c in range(n_classes):
                m_class_samples[c],m_class_borders[c] = split_samples(
                    m_class_samples[c], 2,
                    ratios=[1-annotation_border_ratio,annotation_border_ratio] )

        # Get labeled image for identifying annotations
        if annotation_type.lower() == 'centroids':
            im_label = self.centroids
            im_label_class = self.centroids_typenr
        elif annotation_type.lower() == 'bodies':
            im_label = self.bodies
            im_label_class = self.bodies_typenr

        # Get labeled image for return annotations
        if return_annotations is not False:
            if return_annotations.lower() == 'centroids':
                return_im_label = self.centroids
            elif return_annotations.lower() == 'bodies':
                return_im_label = self.bodies

        # Predefine output matrices
        samples = np.zeros( (m_samples,
            self.n_channels*zoom_size[0]*zoom_size[1]) )
        if return_annotations is not False:
            annotations = np.zeros( (m_samples, zoom_size[0]*zoom_size[1]) )
        labels = np.zeros( (m_samples, n_classes) )
        count = 0

        # Loop over output classes
        for c in range(n_classes):

            # Get image where only border pixels are labeled (either pos or neg)
            if annotation_border_ratio is not None:
                brdr_val = 1 if class_labels[c] == 0 else 0
                im_label_er = ndimage.binary_erosion(
                    ndimage.binary_erosion( im_label_class==class_labels[c],
                        border_value=brdr_val ), border_value=brdr_val )
                im_label_border = im_label_class==class_labels[c]
                im_label_border[im_label_er>0] = 0

            # Get lists of all pixels that fall in one class
            pix_y,pix_x = get_labeled_pixel_coordinates( \
                im_label_class==class_labels[c],
                exclude_border=self.exclude_border_tuple )
            if annotation_border_ratio is not None:
                brdr_pix_y,brdr_pix_x = get_labeled_pixel_coordinates( \
                    im_label_border,
                    exclude_border=self.exclude_border_tuple )

            # Get list of random indices for pixel coordinates
            if len(pix_x) < m_class_samples[c]:
                print("!! Warning: fewer samples of class {} (n={})".format( \
                    c, len(pix_x)) + " than requested (m={})".format(m_class_samples[c]))
                print("   Returning duplicate samples...")
                random_px = np.random.choice( len(pix_x),
                                            m_class_samples[c], replace=True )
            else:
                random_px = np.random.choice( len(pix_x),
                                            m_class_samples[c], replace=False )

            if annotation_border_ratio is not None:
                if len(brdr_pix_x) < m_class_borders[c]:
                    print("!! Warning: fewer border samples of class {} (n={})".format( \
                        c, len(brdr_pix_x)) + " than requested (m={})".format(m_class_borders[c]))
                    print("   Returning duplicate samples...")
                    random_brdr_px = np.random.choice( len(brdr_pix_x),
                                            m_class_borders[c], replace=True )
                else:
                    random_brdr_px = np.random.choice( len(brdr_pix_x),
                                            m_class_borders[c], replace=False )

            # Loop samples
            for p in random_px:
                nr = im_label[pix_y[p], pix_x[p]]
                if not morph_annotations:
                    samples[count,:] = image2vec( zoom( self.channel,
                        pix_y[p], pix_x[p],
                        zoom_size=zoom_size, normalize=normalize_samples ) )
                    if return_annotations and not segment_all:
                        annotations[count,:] = image2vec( zoom( \
                            return_im_label==nr, pix_y[p], pix_x[p],
                            zoom_size=zoom_size, normalize=normalize_samples ) )
                    elif return_annotations and segment_all:
                        annotations[count,:] = image2vec( zoom( \
                            return_im_label>0, pix_y[p], pix_x[p],
                            zoom_size=zoom_size, normalize=normalize_samples ) )
                else:
                    rotation = float(np.random.choice( rotation_list, 1 ))
                    scale = ( float(np.random.choice( scale_list_y, 1 )), \
                                float(np.random.choice( scale_list_x, 1 )) )
                    noise_level = float(np.random.choice( noise_level_list, 1 ))

                    samples[count,:] = image2vec( morphed_zoom( self.channel,
                        pix_y[p], pix_x[p], zoom_size,
                        rotation=rotation, scale_xy=scale,
                        normalize=normalize_samples, noise_level=noise_level ) )
                    if return_annotations and not segment_all:
                        annotations[count,:] = image2vec( morphed_zoom( \
                            (return_im_label==nr).astype(np.float),
                            pix_y[p], pix_x[p], zoom_size,
                            rotation=rotation, scale_xy=scale,
                            normalize=normalize_samples, noise_level=0 ) )
                    elif return_annotations and segment_all:
                        annotations[count,:] = image2vec( morphed_zoom( \
                            (return_im_label>0).astype(np.float),
                            pix_y[p], pix_x[p], zoom_size,
                            rotation=rotation, scale_xy=scale,
                            normalize=normalize_samples, noise_level=0 ) )
                labels[count,c] = 1
                count = count + 1

            # Positive border examples
            if annotation_border_ratio is not None:
                for p in random_brdr_px:
                    nr = im_label[brdr_pix_y[p], brdr_pix_x[p]]
                    if not morph_annotations:
                        samples[count,:] = image2vec( zoom( self.channel,
                            brdr_pix_y[p], brdr_pix_x[p],
                            zoom_size=zoom_size, normalize=normalize_samples ) )
                        if return_annotations and not segment_all:
                            annotations[count,:] = image2vec( zoom( return_im_label==nr,
                                brdr_pix_y[p], brdr_pix_x[p],
                                zoom_size=zoom_size, normalize=normalize_samples ) )
                        elif return_annotations and segment_all:
                            annotations[count,:] = image2vec( zoom( return_im_label>0,
                                brdr_pix_y[p], brdr_pix_x[p],
                                zoom_size=zoom_size, normalize=normalize_samples ) )
                    else:
                        rotation = float(np.random.choice( rotation_list, 1 ))
                        scale = ( float(np.random.choice( scale_list_y, 1 )), \
                                    float(np.random.choice( scale_list_x, 1 )) )
                        noise_level = float(np.random.choice( noise_level_list, 1 ))

                        samples[count,:] = image2vec( morphed_zoom( self.channel,
                            brdr_pix_y[p], brdr_pix_x[p], zoom_size,
                            rotation=rotation, scale_xy=scale,
                            normalize=normalize_samples, noise_level=noise_level ) )
                        if return_annotations and not segment_all:
                            annotations[count,:] = image2vec( morphed_zoom(
                                (return_im_label==nr).astype(np.float),
                                brdr_pix_y[p], brdr_pix_x[p], zoom_size,
                                rotation=rotation, scale_xy=scale,
                                normalize=normalize_samples, noise_level=0 ) )
                        elif return_annotations and segment_all:
                            annotations[count,:] = image2vec( morphed_zoom(
                                (return_im_label>0).astype(np.float),
                                brdr_pix_y[p], brdr_pix_x[p], zoom_size,
                                rotation=rotation, scale_xy=scale,
                                normalize=normalize_samples, noise_level=0 ) )
                    labels[count,c] = 1
                    count = count + 1

        # Return samples, labels, annotations etc
        if return_annotations:
            annotations[annotations<0.5]=0
            annotations[annotations>=0.5]=1
            return samples,labels,annotations
        else:
            return samples,labels,[]

    def generate_cnn_annotations_cb(self, min_size=None, max_size=None,
                dilation_factor_centroids=0, dilation_factor_bodies=0,
                re_dilate_bodies=0 ):
        """Generates annotations from CNN detected bodies. If detected
        centroids are present, it uses those to identify single annotations
        and uses the detected bodies to get the outlines
        min_size:  Minimum number of pixels of the annotations
        max_size:  Maximum number of pixels of the annotations
        dilation_factor_centroids: Dilates or erodes centroids before
                                    segentation(erosion will get rid of
                                    'speccles', dilations won't do much good)
        dilation_factor_bodies:    Dilates or erodes annotation bodies
                                    before segmentation
        re_dilate_bodies:          Dilates or erodes annotation bodies
                                    after segmentation
        """
        # Check if centroids are detected
        if self.detected_centroids is None:
            do_centroids = False
        else:
            do_centroids = True

        detected_bodies = np.array(self.detected_bodies)
        if do_centroids:
            detected_centroids = np.array(self.detected_centroids)

        # Remove annotated pixels too close to the border artifact region
        if self.exclude_border['left'] > 0:
            detected_bodies[ :, :self.exclude_border['left'] ] = 0
        if self.exclude_border['right'] > 0:
            detected_bodies[ :, -self.exclude_border['right']: ] = 0
        if self.exclude_border['top'] > 0:
            detected_bodies[ :self.exclude_border['top'], : ] = 0
        if self.exclude_border['bottom'] > 0:
            detected_bodies[ -self.exclude_border['bottom']:, : ] = 0

        # Dilate or erode centroids
        if do_centroids:
            if dilation_factor_centroids>0:
                for _ in range(dilation_factor_centroids):
                    detected_centroids = \
                        ndimage.binary_dilation(detected_centroids)
            elif dilation_factor_centroids<0:
                for _ in range(-1*dilation_factor_centroids):
                    detected_centroids = \
                        ndimage.binary_erosion(detected_centroids)

        # Dilate or erode bodies
        if dilation_factor_bodies>0:
            for _ in range(dilation_factor_bodies):
                detected_bodies = ndimage.binary_dilation(detected_bodies)
        elif dilation_factor_bodies<0:
            for _ in range(-1*dilation_factor_bodies):
                detected_bodies = ndimage.binary_erosion(detected_bodies)

        # Get rid of centroids that have no bodies associated with them
        if do_centroids:
            detected_centroids[detected_bodies==0] = 0

        # Get labeled centroids and bodies
        if do_centroids:
            centroid_labels = measure.label(detected_centroids, background=0)
            n_centroid_labels = centroid_labels.max()
            print("Found {} putative centroids".format(n_centroid_labels))
        body_labels = measure.label(detected_bodies, background=0)
        n_body_labels = body_labels.max()
        print("Found {} putative bodies".format(n_body_labels))

        # Nothing labeled, no point to continue
        if n_centroid_labels == 0 or n_body_labels == 0:
            print("Aborting ...")
            return 0

        # If only bodies, convert labeled bodies annotations
        if not do_centroids:
            print("Converting labeled body image into annotations {:3d}".format(0),
                    end="", flush=True)
            ann_body_list = []
            for nr in range(1,n_body_labels+1):
                print((3*'\b')+'{:3d}'.format(nr), end='', flush=True)
                body_mask = body_labels==nr
                an_body = Annotation( body_pixels_yx=body_mask)
                ann_body_list.append(an_body)
            print((3*'\b')+'{:3d}'.format(nr))
        else:
            # Convert labeled centroids into centroid and body annotations
            print("Converting labeled centroids and bodies into annotations {:3d}".format(0),
                    end="", flush=True)
            ann_body_list = []
            ann_body_nr_list = []
            ann_centr_list = []
            for nr in range(1,n_centroid_labels+1):
                print((3*'\b')+'{:3d}'.format(nr), end='', flush=True)
                mask = centroid_labels==nr
                an_centr = Annotation( body_pixels_yx=mask)
                ann_centr_list.append(an_centr)

                body_nr = body_labels[int(an_centr.y),int(an_centr.x)]
                ann_body_nr_list.append(body_nr)
                body_mask = body_labels==body_nr
                an_body = Annotation( body_pixels_yx=body_mask)
                ann_body_list.append(an_body)
            print((3*'\b')+'{:3d}'.format(nr))

            # Loop centroid annotations to remove overlap of body annotations
            print("Removing overlap of annotation {:3d}".format(0), end="", flush=True)
            for nr1 in range(len(ann_centr_list)):
                print((3*'\b')+'{:3d}'.format(nr1), end='', flush=True)

                # Find out if the centroid shares the body with another centroid
                shared_list = []
                for nr2 in range(len(ann_centr_list)):
                    if (ann_body_nr_list[nr1] == ann_body_nr_list[nr2]) \
                        and (ann_body_nr_list[nr1] > 0):
                        shared_list.append(nr2)

                # If more than one centroid owns the same body, split it
                if len(shared_list) > 1:

                    # for each pixel, calculate the distance to each centroid
                    D = np.zeros((ann_body_list[nr1].body.shape[0],len(shared_list)))
                    for n,c in enumerate(shared_list):
                        cy, cx = ann_centr_list[c].y, ann_centr_list[c].x
                        for p,(y,x) in enumerate(ann_body_list[c].body):
                            D[p,n] = np.sqrt( ((cy-y)**2) + ((cx-x)**2) )

                    # Find the closest centroid for each pixel
                    closest_cntr = np.argmin(D,axis=1)

                    # For each centroid, get a new annotation with closest pixels
                    for n,c in enumerate(shared_list):
                        B = ann_body_list[c].body[closest_cntr==n,:]
                        new_ann = Annotation(body_pixels_yx=B)
                        ann_body_nr_list[c] = 0
                        ann_body_list[c] = new_ann
            print((3*'\b')+'{:3d}'.format(nr1))

        # Remove too small annotations
        if min_size is not None:
            remove_ix = []
            for nr in range(len(ann_body_list)):
                if ann_body_list[nr].body.shape[0] < min_size:
                    remove_ix.append(nr)
            if len(remove_ix) > 0:
                print("Removing {} annotations where #pixels < {}".format(
                    len(remove_ix), min_size))
            for ix in reversed(remove_ix):
                del ann_body_list[ix]

        # Remove too large annotations
        if max_size is not None:
            remove_ix = []
            for nr in range(len(ann_body_list)):
                if ann_body_list[nr].body.shape[0] > max_size:
                    remove_ix.append(nr)
            if len(remove_ix) > 0:
                print("Removing {} annotations where #pixels > {}".format(
                    len(remove_ix), max_size))
            for ix in reversed(remove_ix):
                del ann_body_list[ix]

        # Dilate or erode annotated bodies
        if re_dilate_bodies != 0:
            print("Dilating annotated bodies by a factor of {}: {:3d}".format(
                re_dilate_bodies,0), end="", flush=True)
            for nr in range(len(ann_body_list)):
                print((3*'\b')+'{:3d}'.format(nr+1), end='', flush=True)
                masked_image = np.zeros(self.detected_bodies.shape)
                ann_body_list[nr].mask_body(
                    image=masked_image, dilation_factor=re_dilate_bodies)
                ann_body_list[nr] = Annotation( body_pixels_yx=masked_image)
            print((3*'\b')+'{:3d}'.format(nr+1))

        # Set the internal annotation list
        self.annotation = ann_body_list


    def image_grid_RGB( self, image_size, image_type='image', annotation_nrs=None,
                        n_x=10, n_y=6, channel_order=(0,1,2),
                        normalize_samples=False, auto_scale=False,
                        amplitude_scaling=(1.33,1.33,1), line_color=0 ):
        """ Constructs a 3d numpy.ndarray tiled with a grid of RGB images from
            the annotations. If more images are requested than can be tiled,
            it chooses and displays a random subset.
        image_size:        2 dimensional size of the zoom-images (y,x)
        image_type:        'image', 'bodies', 'centroids'
        annotation_nrs:    List with nr of the to be displayed annotations
        n_x:               Number of images to show on x axis of grid
        n_y:               Number of images to show on y axis of grid
        channel_order:     Tuple indicating which channels are R, G and B
        auto_scale:        Scale each individual image to its maximum (T/F)
        normalize_samples: Scale each individual channel to its maximum
        amplitude_scaling: Intensity scaling of each color channel
        line_color:        Intensity (gray scale) of line between images
        Returns numpy.ndarray (y,x,RGB) and a list with center_shifts (y,x)
        """

        # Get indices of images to show
        if annotation_nrs is None:
            annotation_nrs = list(range(self.n_annotations))
        n_images = len(annotation_nrs)
        if n_images <= n_x*n_y:
            im_ix = list(range(n_images))
        else:
            im_ix = np.random.choice( n_images, n_x*n_y, replace=False )

        # Get coordinates of where images will go
        y_coords = []
        offset = 0
        for i in range(n_y):
            offset = i * (image_size[0] + 1)
            y_coords.append(offset+np.array(range(image_size[0])))
        max_y = np.max(y_coords[i]) + 1

        x_coords = []
        offset = 0
        for i in range(n_x):
            offset = i * (image_size[1] + 1)
            x_coords.append(offset+np.array(range(image_size[1])))
        max_x = np.max(x_coords[i]) + 1

        rgb_coords = np.array(list(range(3)))

        # Fill grid
        im_count = 0
        rgb_im = np.zeros((image_size[0],image_size[1],3))
        grid = np.zeros((max_y,max_x,3))+line_color
        center_shift = []
        for y in range(n_y):
            for x in range(n_x):
                if im_count < n_images:
                    for ch in range(3):
                        if image_type.lower() == 'image':
                            im = self.channel[channel_order[ch]]
                        if image_type.lower() == 'centroids':
                            im = self.centroids>0.5
                        if image_type.lower() == 'bodies':
                            im = self.bodies>0.5
                        rgb_im[:,:,ch] = zoom( im,
                            self.annotation[im_count].y,
                            self.annotation[im_count].x, image_size,
                            normalize=normalize_samples, pad_value=0 )
                    if auto_scale:
                        rgb_im = rgb_im / rgb_im.max()
                    grid[np.ix_(y_coords[y],x_coords[x],rgb_coords)] = rgb_im
                    center_shift.append( \
                        ( y_coords[y][0] + (0.5*image_size[0]) -0.5,
                          x_coords[x][0] + (0.5*image_size[0]) -0.5 ) )
                else:
                    break
                im_count += 1
        return grid, center_shift

########################################################################
### Class AnnotatedImageSet
########################################################################

class AnnotatedImageSet(object):
    """Class that represents a dataset of annotated images and organizes
    the dataset for feeding in machine learning algorithms"""

    def __init__(self, downsample=None):
        """Initializes
            downsample:          Downsample to be imported images, borders
                                 and ROI's by a certain factor
        """
        # initializes the list of annotated images
        self._downsample = downsample
        self.ai_list = []
        self._body_dilation_factor = 0
        self._centroid_dilation_factor = 0
        self._include_annotation_typenrs = None
        self._n_channels = 0

    def __str__(self):
        return "AnnotatedImageSet (# Annotated Images = {:.0f}" \
                ")".format(self.n_annot_images)

    # **********************************
    # *****  Read only properties  *****
    @property
    def n_annot_images(self):
        return len(self.ai_list)

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def downsamplingfactor(self):
        """Returns the (read-only) downsampling factor"""
        return self._downsample

    # ********************************************
    # *****  Handling the annotation typenr  *****
    @property
    def class_labels(self):
        """Returns the class labels that are set for training"""
        class_labels = [0,]
        class_labels.extend(list(self.include_annotation_typenrs))
        return class_labels

    @property
    def include_annotation_typenrs(self):
        """Returns the annotation typenrs"""
        return self._include_annotation_typenrs

    @include_annotation_typenrs.setter
    def include_annotation_typenrs(self, annotation_typenrs):
        """Updates the internal annotation typenr if not equal to last set nrs
        """
        if isinstance(annotation_typenrs,int):
            annotation_typenrs = set([annotation_typenrs,])
        elif annotation_typenrs is None:
            pass
        else:
            annotation_typenrs = set(annotation_typenrs)

        if isinstance(annotation_typenrs,set):
            if 0 in annotation_typenrs:
                annotation_typenrs.remove(0)

        if annotation_typenrs != self._include_annotation_typenrs:
            new_annotation_type_nrs = set()
            for nr in range(self.n_annot_images):
                if self.ai_list[nr].include_annotation_typenrs != annotation_typenrs:
                    self.ai_list[nr].include_annotation_typenrs = annotation_typenrs
                if annotation_typenrs is None:
                    new_annotation_type_nrs.update(self.ai_list[nr].include_annotation_typenrs)
            if annotation_typenrs is not None:
                self._include_annotation_typenrs = annotation_typenrs
            else:
                self._include_annotation_typenrs = new_annotation_type_nrs


    # ********************************************
    # *****  Handling cropping of annot-ims  *****
    def crop( self, left, top, width, height ):
        """Crops the image channels, annotations and borders
        left:   Left most pixel in cropped image (0 based)
        top:    Top most pixel in cropped image (0 based)
        width:  Width of cropped region
        height: Height of cropped region
        """
        for nr in range(self.n_annot_images):
            self.ai_list[nr].crop(left, top, width, height )

    # *******************************************
    # *****  Handling the annotated bodies  *****
    @property
    def body_dilation_factor(self):
        """Returns the body dilation factor"""
        return(self._body_dilation_factor)

    @body_dilation_factor.setter
    def body_dilation_factor(self, dilation_factor):
        """Updates the internal body annotation mask with dilation_factor"""
        if dilation_factor != self._body_dilation_factor:
            for nr in range(self.n_annot_images):
                self.ai_list[nr].body_dilation_factor = dilation_factor
            self._body_dilation_factor = dilation_factor

    # **********************************************
    # *****  Handling the annotated centroids  *****
    @property
    def centroid_dilation_factor(self):
        """Returns the centroid dilation factor"""
        return(self._centroid_dilation_factor)

    @centroid_dilation_factor.setter
    def centroid_dilation_factor(self, dilation_factor):
        """Updates the internal centroid annotation mask with dilation_factor"""
        if dilation_factor != self._centroid_dilation_factor:
            for nr in range(self.n_annot_images):
                self.ai_list[nr].centroid_dilation_factor = dilation_factor
            self._centroid_dilation_factor = dilation_factor

    # ********************************************
    # *****  Produce training/test data set  *****
    def data_sample(self, zoom_size, annotation_type='Bodies',
            m_samples=100, return_annotations=False,
            sample_ratio=None, annotation_border_ratio=None,
            normalize_samples=False, segment_all=False,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None ):
        """Constructs a random sample of with linearized annotation data,
            organized in a 2d matrix (m samples x n pixels) half of which is
            from within an annotation, and half from outside. It takes equal
            amounts of data from each annotated image in the list.
            zoom_size:         2 dimensional size of the image (y,x)
            annotation_type:   'Bodies' or 'Centroids'
            m_samples:         number of training samples
            return_annotations:  Returns annotations in addition to
                                 samples and labels. If False, returns empty
                                 list. Otherwise set to 'Bodies' or 'Centroids'
            sample_ratio:        List with ratio of samples per groups (sum=1)
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            normalize_samples: Scale each individual channel to its maximum
            segment_all:       Segments all instead of single annotations (T/F)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            Returns tuple with samples as 2d numpy matrix, labels as
            2d numpy matrix and if requested annotations as 2d numpy matrix
            or otherwise an empty list as third item"""

        # Get number of classes
        n_classes = len(self.class_labels)

        # Calculate number of pixels in linearized image
        n_pix_lin = self.ai_list[0].n_channels * zoom_size[0] * zoom_size[1]

        # List with start and end sample per AnnotatedImage
        m_set_samples_list = np.round( np.linspace( 0, m_samples,
                                                    self.n_annot_images+1 ) )

        # Predefine output matrices
        samples = np.zeros( (m_samples, n_pix_lin) )
        if return_annotations is not False:
            annotations = np.zeros( (m_samples, zoom_size[0]*zoom_size[1]) )
        else:
            annotations = []
        labels = np.zeros( (m_samples, n_classes) )

        # Loop AnnotatedImages
        for s in range(self.n_annot_images):

            # Number of samples for this AnnotatedImage
            m_set_samples = int(m_set_samples_list[s+1]-m_set_samples_list[s])

            # Get samples, labels, annotations
            s_samples,s_labels,s_annotations = \
                self.ai_list[s].get_batch(
                    zoom_size, annotation_type=annotation_type,
                    m_samples=m_set_samples,
                    return_annotations=return_annotations,
                    sample_ratio=sample_ratio,
                    annotation_border_ratio=annotation_border_ratio,
                    normalize_samples=normalize_samples, segment_all=segment_all,
                    morph_annotations=morph_annotations,
                    rotation_list=rotation_list, scale_list_x=scale_list_x,
                    scale_list_y=scale_list_y, noise_level_list=noise_level_list )

            # put samples, labels and possibly annotations in
            samples[int(m_set_samples_list[s]):int(m_set_samples_list[s+1]),:] \
                = s_samples
            labels[int(m_set_samples_list[s]):int(m_set_samples_list[s+1]),:] \
                = s_labels
            if return_annotations is not False:
                annotations[int(m_set_samples_list[s]):int(m_set_samples_list[s+1]),:] \
                    = s_annotations
        return samples,labels,annotations

    # **************************************
    # *****  Load data from directory  *****
    def load_data_dir_tiff_mat(self, data_directory,
                    normalize=True, use_channels=None, exclude_border=None):
        """Loads all Tiff images or *channel.mat and accompanying ROI.mat
        files from a single directory that contains matching sets of .tiff
        or *channel.mat and .mat files
        data_directory:  path
        normalize:  Normalize to maximum of image
        use_channels:  tuple holding channel numbers/order to load (None=all)
        exclude_border: Load border exclude region from file
        """
        # Get list of all .tiff file and .mat files
        image_files = glob.glob(path.join(data_directory,'*channels.mat'))
        if len(image_files) == 0:
            image_files = glob.glob(path.join(data_directory,'*.tiff'))
        mat_files = glob.glob(path.join(data_directory,'*ROI*.mat'))

        # Exclude border files
        if isinstance(exclude_border,str):
            if exclude_border.lower() == 'load':
                brdr_files = glob.glob(path.join(data_directory,'*Border*.mat'))

        # Loop files and load images and annotations
        print("\nLoading image and annotation files:")
        annotation_type_nrs = set()
        for f, (image_file, mat_file) in enumerate(zip(image_files,mat_files)):
            image_filepath, image_filename = path.split(image_file)
            mat_filepath, mat_filename = path.split(mat_file)
            print("{:2.0f}) {} -- {}".format(f+1,image_filename,mat_filename))

            # Create new AnnotatedImage, add images and annotations
            anim = AnnotatedImage(downsample=self.downsamplingfactor)
            if self.include_annotation_typenrs is not None:
                anim.include_annotation_typenrs = self.include_annotation_typenrs
            anim.add_image_from_file( image_filename, image_filepath,
                            normalize=normalize, use_channels=use_channels )
            anim.import_annotations_from_mat( mat_filename, mat_filepath )

            if isinstance(exclude_border,str):
                if exclude_border.lower() == 'load':
                    anim.exclude_border = brdr_files[f]
            if isinstance(exclude_border,list) \
                    or isinstance(exclude_border,tuple):
                anim.exclude_border = exclude_border

            # Check if the number of channels is the same
            if len(self.ai_list) == 0:
                self._n_channels = anim.n_channels
            else:
                if self._n_channels != anim.n_channels:
                    print("!!! CRITICAL WARNING !!!")
                    print("-- Number of channels is not equal for all annotated images --")

            # Append AnnotatedImage to the internal list
            print("    - "+anim.__str__())
            self.ai_list.append(anim)
            annotation_type_nrs.update(anim.include_annotation_typenrs)
        if self.include_annotation_typenrs is None:
            self.include_annotation_typenrs = annotation_type_nrs
