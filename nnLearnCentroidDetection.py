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

data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet'
zoom_size = (36,36)
ais = ia.AnnotatedImageSet()
ais.load_data_dir(data_path)


########################################################################
# Set up network
nn = cn.ConvNetCnv2Fc1( input_image_size=zoom_size, n_input_channels=3,
                        output_size=(1,2) )
print(nn)

# ########################################################################
# # Train for 100 steps
# start = time.time()
# for t in range(3000):
#     if t % 10 != 0: # Just train
#         samples,labels = get_roi_training_data( im0_norm, im1_norm, im2_norm,
#                                                 masked_roi_im, zoom_size, 200 )
#         sess.run(train_step,
#             feed_dict={x: samples, y_: labels, keep_prob: 0.5})
#         print('.', end="", flush=True)
#     else: # Dipslay progress
#         samples,labels = get_roi_training_data( im0_norm, im1_norm, im2_norm,
#                                                 masked_roi_im, zoom_size, 200 )
#         result = sess.run([merged_summary,accuracy],
#             feed_dict={x: samples, y_: labels, keep_prob: 1.0})
#         summary_str = result[0]
#         acc = result[1]
#         writer.add_summary(summary_str, t)
#         end = time.time()
#         print('Step {:4d}: Acc = {:6.4f} (t={})'.format(
#                 t, acc, str(datetime.timedelta(seconds=np.round(end-start))) ))
#         save_path = saver.save(sess, "/tmp/roi_conv_model3.ckpt")
#         print('  -> Model saved in file: {}'.format(save_path))
