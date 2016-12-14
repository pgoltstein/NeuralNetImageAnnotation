#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:01:22 2016

@author: pgoltstein
"""

class ROI(object):
    """Class that holds individual ROI data"""

    def __init__(self,body_pixels_yx):
        # Store body pixels
        self.body = body_pixels_yx

        # Calculate centroids
        self.y = np.mean(self.body[:,0])
        self.x = np.mean(self.body[:,1])

        # Calculate perimenter

    def
