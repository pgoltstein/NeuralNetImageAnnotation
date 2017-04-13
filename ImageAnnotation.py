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


########################################################################
### Functions
########################################################################

def zoom( image, y, x, zoom_size, pad_value=0 ):
    """Crops a(n) (list of) image(s) to the area of tuple/list zoom_size
    around the supplied y, x coordinates. Pads out of range values.
    image:      Single 2d numpy.ndarray or list of 2d numpy.ndarrays
    y, x:       Center coordinates
    zoom_size:  Size of zoomed image (y,x)
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
        max_ix_exceed = np.max((np.abs(np.min(ix_y)), np.abs(np.min(ix_x)),
                    np.max(ix_y)-image.shape[0], np.max(ix_x)-image.shape[1] ))
        if max_ix_exceed > 0:
            image_temp = np.zeros((image.shape+max_ix_exceed+1))+pad_value
            image_temp[0:image.shape[0],0:image.shape[1]] = image
            return image_temp[ np.ix_(ix_y,ix_x) ]
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
            image = image + noise_mask
        return image

def morphed_zoom( image, y, x, zoom_size, pad_value=0,
                    rotation=0, scale_xy=(1,1), noise_level=0 ):
    """Crops image or image list to area of zoom_size around centroid
    image:        Single 2d numpy.ndarray or list of 2d numpy.ndarrays
    y, x:         Center coordinates
    zoom_size:    (y size, x size)
    pad_value:    Value for out of range coordinates
    rotation:     Rotation of annotation in degrees (0-360 degrees)
    scale_xy:     Determines fractional scaling on x/y axis.
                  Min-Max = (0.5,0.5) - (2,2)
    noise_level:  Level of random noise
    returns tuple holding (morped_zoom, morped_annotation)"""
    im = zoom( image=image, y=y, x=x,
        zoom_size=(zoom_size[0]*3,zoom_size[1]*3), pad_value=pad_value )
    im = morph( image=im,
            rotation=rotation, scale_xy=scale_xy, noise_level=noise_level )
    if isinstance( im, list ):
        y_pos, x_pos = (im[0].shape[0]-1)/2, (im[0].shape[1]-1)/2
    else:
        y_pos, x_pos = (im.shape[0]-1)/2, (im.shape[1]-1)/2
    return zoom( im, y=y_pos, x=x_pos,
                     zoom_size=zoom_size, pad_value=pad_value )

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


########################################################################
### Class Annotation
########################################################################

class Annotation(object):
    """Class that holds an individual image annotation"""

    def __init__( self, body_pixels_yx, annotation_name="",
                                    type_nr=1, group_nr=0):
        """Initialize.
            body_pixels_yx: list/tuple of (y,x) coordinates
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

    def zoom(self, image, zoom_size, pad_value=0 ):
        """Crops image to area of tuple/list zoom_size around centroid
        image:      Single 2d numpy.ndarray
        zoom_size:  (y size, x size)
        pad_value:  Value for out of range coordinates
        returns zoomed image"""
        return zoom( image=image, y=self._y, x=self._x,
                        zoom_size=zoom_size, pad_value=pad_value )

    def morphed_zoom(self, image, zoom_size, pad_value=0,
                        rotation=0, scale_xy=(1,1), noise_level=0 ):
        """Crops image to area of tuple/list zoom_size around centroid
        image:        Single 2d numpy.ndarray
        zoom_size:    (y size, x size)
        pad_value:  Value for out of range coordinates
        rotation:     Rotation of annotation in degrees (0-360 degrees)
        scale_xy:     Determines fractional scaling on x/y axis.
                      Min-Max = (0.5,0.5) - (2,2)
        noise_level:  Level of random noise
        returns tuple holding (morped_zoom, morped_annotation)"""
        return morphed_zoom( image, self._y, self._x, zoom_size=zoom_size,
                    pad_value=pad_value, rotation=rotation,
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

    def __init__( self, image_data=None, annotation_data=None ):
        """Initialize image list and channel list
            channel:     list or tuple of same size images
            annotation:  list or tuple of Annotation objects"""
        self._bodies = None
        self._body_dilation_factor = 0
        self._centroids = None
        self._centroid_dilation_factor = 0
        self._y_res = 0
        self._x_res = 0
        self._channel = []
        self._annotation = []
        if image_data is not None:
            self.channel = image_data
        if annotation_data is not None:
            self.annotation = annotation_data

    def __str__(self):
        return "AnnotatedImage (#channels={:.0f}, #annotations={:.0f}" \
                ")".format(self.n_channels, self.n_annotations)

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
                self._channel.append( np.array(im) )
        else:
            self._channel.append( np.array(image_data) )
        self._y_res,self._x_res = self._channel[0].shape
        # Update masks if there are annotations and the image resolution changed
        if self.n_annotations > 0 and ( (y_res_old != self.y_res)
                                    or (x_res_old != self.x_res) ):
            self._set_bodies()
            self._set_centroids()


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
                    self._channel.append(im_x)
            else:
                if normalize:
                    im = im / im.max()
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
        for an in annotation_data:
            self._annotation.append( Annotation(
                body_pixels_yx=an.body,
                annotation_name=an.name,
                type_nr=an.type_nr,
                group_nr=an.group_nr) )
        # Update masks if there is at least one image channel
        if self.n_channels > 0:
            self._set_bodies()
            self._set_centroids()

    def import_annotations_from_mat(self, file_name, file_path='.'):
        """Reads data from ROI.mat file and fills the annotation_list. Can
        handle .mat files holding either a ROI or a ROIpy structure
        file_name:     String holding name of ROI file
        file_path:     String holding file path"""

        # Load mat file with ROI data
        mat_data = loadmat(path.join(file_path,file_name))
        annotation_list = []
        if 'ROI' in mat_data.keys():
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
        elif 'ROIpy' in mat_data.keys():
            nROIs = len(mat_data['ROIpy'][0])
            for c in range(nROIs):
                body = mat_data['ROIpy'][0][c]['body'][0][0]
                body = np.array([body[:,1],body[:,0]]).transpose()
                body = body-1 # Matlab (1-index) to Python (0-index)
                type_nr = int(mat_data['ROIpy'][0][c]['type'][0][0][0][0])
                name = str(mat_data['ROIpy'][0][c]['typename'][0][0][0])
                group_nr = int(mat_data['ROIpy'][0][c]['group'][0][0][0][0])
                annotation_list.append( Annotation( body_pixels_yx=body,
                        annotation_name=name, type_nr=type_nr, group_nr=group_nr ) )
        self.annotation = annotation_list

    def export_annotations_to_mat(self, file_name, file_path='.'):
        """Writes annotations to ROI_py.mat file
        file_name:  String holding name of ROI file
        file_path:  String holding file path"""
        roi_list = []
        for nr in range(self.n_annotations):
            roi_dict = {}
            roi_dict['nr']=nr
            roi_dict['group']=self.annotation[nr].group_nr
            roi_dict['type']=self.annotation[nr].type_nr
            roi_dict['typename']=self.annotation[nr].name
            # Mind the +1: Matlab (1-index) to Python (0-index)
            roi_dict['x']=self.annotation[nr].x+1
            roi_dict['y']=self.annotation[nr].y+1
            roi_dict['size']=self.annotation[nr].size
            roi_dict['perimeter']=self.annotation[nr].perimeter+1
            body = np.array([self.annotation[nr].body[:,1],
                             self.annotation[nr].body[:,0]]).transpose()+1
            roi_dict['body']=body
            roi_list.append(roi_dict)
        savedata = {}
        savedata['ROIpy']=roi_list
        savemat(path.join(file_path,file_name),savedata)

    # ******************************************
    # *****  Handling the annotated bodies *****
    @property
    def bodies(self):
        """Returns an image with annotation bodies masked"""
        return self._bodies

    def _set_bodies(self):
        """Sets the internal body annotation mask with specified parameters"""
        self._bodies = np.zeros_like(self._channel[0])
        for nr in range(self.n_annotations):
            self._annotation[nr].mask_body(self._bodies,
                dilation_factor=self._body_dilation_factor,
                mask_value=nr+1, keep_centroid=True)

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

    def _set_centroids(self):
        """Sets the internal centroids annotation mask with specified
        parameters"""
        self._centroids = np.zeros_like(self._channel[0])
        for nr in range(self.n_annotations):
            self._annotation[nr].mask_centroid(self._centroids,
                dilation_factor=self._centroid_dilation_factor,
                mask_value=nr+1)

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
        print("Loaded AnnotatedImage from: {}".format(
                                    path.join(file_path,file_name)))

    def save(self,file_name,file_path='.'):
        """Saves image and annotations to .npy file"""
        combined_annotated_image = {}
        combined_annotated_image['image_data'] = self.channel
        combined_annotated_image['annotation_data'] = self.annotation
        np.save(path.join(file_path,file_name), combined_annotated_image)
        print("Saved AnnotatedImage as: {}".format(
                                    path.join(file_path,file_name)))

    # ************************************************
    # *****  Generate NN training/test data sets *****
    def get_batch( self, zoom_size, annotation_type='Bodies',
            m_samples=100, exclude_border=(0,0,0,0), return_annotations=False,
            pos_sample_ratio=0.5, morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None ):
        """Constructs a 2d matrix (m samples x n pixels) with linearized data
            half of which is from within an annotation, and half from outside
            zoom_size:         2 dimensional size of the image (y,x)
            annotation_type:   'Bodies' or 'Centroids'
            m_samples:         number of training samples
            exclude_border:    exclude annotations that are a certain distance
                               to each border. Pix from (left, right, up, down)
            return_annotations:  Returns annotations in addition to
                                 samples and labels. If False, returns empty
                                 list. Otherwise set to 'Bodies' or 'Centroids'
            pos_sample_ratio:  Ratio of positive to negative samples (0.5=
                               equal, 1=only positive samples)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            Returns tuple with samples as 2d numpy matrix, labels as
            2d numpy matrix and if requested annotations as 2d numpy matrix
            or otherwise an empty list as third item"""

        # Calculate number of positive and negative samples
        m_samples_pos = np.int16( m_samples * pos_sample_ratio )
        m_samples_neg = m_samples - m_samples_pos

        # randomly add 1 sample to compensate for rounding down positive samples
        if np.random.rand(1) < ((float(m_samples)*pos_sample_ratio) % 1):
            m_samples_pos = m_samples_pos + 1
            m_samples_neg = m_samples_neg - 1

        # Get lists with coordinates of pixels within and outside of centroids
        (pix_x,pix_y) = np.meshgrid(np.arange(self.y_res),np.arange(self.x_res))
        if annotation_type.lower() == 'centroids':
            im_label = self.centroids
        elif annotation_type.lower() == 'bodies':
            im_label = self.bodies
        if return_annotations is not False:
            if return_annotations.lower() == 'centroids':
                return_im_label = self.centroids
            elif return_annotations.lower() == 'bodies':
                return_im_label = self.bodies

        roi_positive_x = pix_x.ravel()[im_label.ravel() > 0.5]
        roi_positive_y = pix_y.ravel()[im_label.ravel() > 0.5]
        roi_negative_x = pix_x.ravel()[im_label.ravel() == 0]
        roi_negative_y = pix_y.ravel()[im_label.ravel() == 0]

        # Exclude all pixels that are within half-zoom from the border
        roi_positive_inclusion = \
            np.logical_and( np.logical_and( np.logical_and(
                roi_positive_x>exclude_border[0],
                roi_positive_x<(self.x_res-exclude_border[1]) ),
                roi_positive_y>exclude_border[2] ),
                roi_positive_y<(self.y_res-exclude_border[3]) )
        roi_positive_x = roi_positive_x[ roi_positive_inclusion ]
        roi_positive_y = roi_positive_y[ roi_positive_inclusion ]
        roi_negative_inclusion = \
            np.logical_and( np.logical_and( np.logical_and(
                roi_negative_x>exclude_border[0],
                roi_negative_x<(self.x_res-exclude_border[1]) ),
                roi_negative_y>exclude_border[2] ),
                roi_negative_y<(self.y_res-exclude_border[3]) )
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
        if return_annotations is not False:
            annotations = np.zeros( (m_samples, zoom_size[0]*zoom_size[1]) )
        labels = np.zeros( (m_samples, 2) )
        count = 0

        # Positive examples
        for p in random_pos:
            nr = im_label[roi_positive_y[p], roi_positive_x[p]]
            if not morph_annotations:
                samples[count,:] = image2vec( zoom( self.channel,
                    roi_positive_y[p], roi_positive_x[p], zoom_size=zoom_size ) )
                if return_annotations:
                    annotations[count,:] = image2vec( zoom( return_im_label==nr,
                        roi_positive_y[p], roi_positive_x[p],
                        zoom_size=zoom_size ) )
            else:
                rotation = float(np.random.choice( rotation_list, 1 ))
                scale = ( float(np.random.choice( scale_list_y, 1 )), \
                            float(np.random.choice( scale_list_x, 1 )) )
                noise_level = float(np.random.choice( noise_level_list, 1 ))

                samples[count,:] = image2vec( morphed_zoom( self.channel,
                    roi_positive_y[p], roi_positive_x[p], zoom_size,
                    rotation=rotation, scale_xy=scale, noise_level=noise_level ) )
                if return_annotations:
                    annotations[count,:] = image2vec( morphed_zoom( return_im_label==nr,
                        roi_positive_y[p], roi_positive_x[p], zoom_size,
                        rotation=rotation, scale_xy=scale, noise_level=noise_level ) )
            labels[count,1] = 1
            count = count + 1

        # Negative examples
        for p in random_neg:
            nr = im_label[roi_negative_y[p], roi_negative_x[p]]
            if not morph_annotations:
                samples[count,:] = image2vec( zoom( self.channel,
                    roi_negative_y[p], roi_negative_x[p], zoom_size=zoom_size ) )
                if return_annotations:
                    annotations[count,:] = image2vec( zoom( return_im_label==nr,
                        roi_negative_y[p], roi_negative_x[p], zoom_size=zoom_size ) )
            else:
                rotation = float(np.random.choice( rotation_list, 1 ))
                scale = ( float(np.random.choice( scale_list_y, 1 )), \
                            float(np.random.choice( scale_list_x, 1 )) )
                noise_level = float(np.random.choice( noise_level_list, 1 ))

                samples[count,:] = image2vec( morphed_zoom( self.channel,
                    roi_negative_y[p], roi_negative_x[p], zoom_size,
                    rotation=rotation, scale_xy=scale, noise_level=noise_level ) )
                if return_annotations:
                    annotations[count,:] = image2vec( morphed_zoom( return_im_label==nr,
                        roi_negative_y[p], roi_negative_x[p], zoom_size,
                        rotation=rotation, scale_xy=scale, noise_level=noise_level ) )
            labels[count,0] = 1
            count = count + 1
        if return_annotations:
            annotations[annotations<0.5]=0
            annotations[annotations>=0.5]=1
            return samples,labels,annotations
        else:
            return samples,labels,[]

    def image_grid_RGB( self, image_size, image_type='image', annotation_nrs=None,
                        n_x=10, n_y=6, channel_order=(0,1,2), auto_scale=False,
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
                            self.annotation[im_count].x,
                            image_size, pad_value=0 )
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

    def __init__(self):
        # initializes the list of annotated images
        self.ai_list = []
        self._body_dilation_factor = 0
        self._centroid_dilation_factor = 0
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
            m_samples=100, exclude_border=(0,0,0,0), return_annotations=False,
            pos_sample_ratio=0.5, morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None ):
        """Constructs a random sample of with linearized annotation data,
            organized in a 2d matrix (m samples x n pixels) half of which is
            from within an annotation, and half from outside. It takes equal
            amounts of data from each annotated image in the list.
            zoom_size:         2 dimensional size of the image (y,x)
            annotation_type:   'Bodies' or 'Centroids'
            m_samples:         number of training samples
            exclude_border:    exclude annotations that are a certain distance
                               to each border. Pix from (left, right, up, down)
            return_annotations:  Returns annotations in addition to
                                 samples and labels. If False, returns empty
                                 list. Otherwise set to 'Bodies' or 'Centroids'
            pos_sample_ratio:  Ratio of positive to negative samples (0.5=
                               equal, 1=only positive samples)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            Returns tuple with samples as 2d numpy matrix, labels as
            2d numpy matrix and if requested annotations as 2d numpy matrix
            or otherwise an empty list as third item"""

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
        labels = np.zeros( (m_samples, 2) )

        # Loop AnnotatedImages
        for s in range(self.n_annot_images):

            # Number of samples for this AnnotatedImage
            m_set_samples = int(m_set_samples_list[s+1]-m_set_samples_list[s])

            # Get samples, labels, annotations
            s_samples,s_labels,s_annotations = \
                self.ai_list[s].get_batch(
                    zoom_size, annotation_type=annotation_type,
                    m_samples=m_set_samples, exclude_border=exclude_border,
                    return_annotations=return_annotations,
                    pos_sample_ratio=pos_sample_ratio,
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
                                normalize=True, use_channels=None):
        """Loads all Tiff images or *channel.mat and accompanying ROI.mat
        files from a single directory that contains matching sets of .tiff
        or *channel.mat and .mat files
        data_directory:  path
        normalize:  Normalize to maximum of image
        use_channels:  tuple holding channel numbers/order to load (None=all)
        """
        # Get list of all .tiff file and .mat files
        image_files = glob.glob(path.join(data_directory,'*channels.mat'))
        if len(image_files) == 0:
            image_files = glob.glob(path.join(data_directory,'*.tiff'))
        mat_files = glob.glob(path.join(data_directory,'*ROI*.mat'))

        # Loop files and load images and annotations
        print("\nLoading image and annotation files:")
        for f, (image_file, mat_file) in enumerate(zip(image_files,mat_files)):
            image_filepath, image_filename = path.split(image_file)
            mat_filepath, mat_filename = path.split(mat_file)
            print("{:2.0f}) {} -- {}".format(f+1,image_filename,mat_filename))

            # Create new AnnotatedImage, add images and annotations
            anim = AnnotatedImage()
            anim.add_image_from_file( image_filename, image_filepath,
                            normalize=normalize, use_channels=use_channels )
            anim.import_annotations_from_mat( mat_filename, mat_filepath )

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
