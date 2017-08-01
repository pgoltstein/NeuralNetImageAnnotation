#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 27, 2017

Contains functions that set up a convolutional neural net for image segmentation

@author: pgoltstein
"""


########################################################################
### Imports
########################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time, datetime
import os
import ImageAnnotation as ia


########################################################################
### Base neural network class
########################################################################

class NeuralNetSegmentation(object):
    """Base class holding a neural network for annotating (multi channel)
    images with multi pixel output (but single class clasification)
    """

    def __init__(self):
        """Does nothing, initialization should be done by subclasses
        List of required subclass variables:

        self.logging:               (True/False)
        self.network_path:          Path
        self.y_res:                 y resolution of input image patches
        self.x_res:                 x resolution of input image patches
        self.n_input_channels:      Number of input channels
        self.y_res_out:             y resolution of output image patches
        self.x_res_out:             x resolution of output image patches
        self.n_output_pixels:       Number of output pixels
        self.fc_dropout:           Dropout fraction of network during training
        self.alpha:                 Learning rate

        self.x:                     Place holder for input data
        self.y_trgt:                Place holder for training target
        self.fc_out_lin:            Output layer
        self.fc_keep_prob:         Place holder for dropout value
        self.train_step:            Optimizer OP
        self.network_prediction:    Prediction of network
        self.accuracy:              Prediction accuracy of network
        self.saver:                 Saver OP

        self.n_samples_trained
        self.n_samples_list
        self.accuracy_list
        self.precision_list
        self.recall_list
        self.F1_list

        List of required subclass methods:

        def display_network_architecture(self):
            Should display the network architecture

        def save_network_architecture(self,network_path):
            Should saves the network architecture into the network path

        """

    def start(self):
        """Initializes all variables and starts session"""
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

    def close(self):
        """Closes the session"""
        self.sess.close()

    def save(self):
        """Saves network architecture and parameters to network path"""
        self.save_network_architecture( network_path=self.network_path )
        self.save_network_parameters(
            file_name='net_parameters', file_path=self.network_path )

    def restore(self):
        """Restores network parameters to last saved values"""
        if os.path.isfile( \
            os.path.join(self.network_path,'net_parameters.nnprm.index')):
            self.load_network_parameters(
                file_name='net_parameters', file_path=self.network_path)
        else:
            self.log("Could not load previous network parameters from:\n{}".format(\
                os.path.join(self.network_path,'net_parameters.nnprm') ))
            self.log("Starting with untrained parameters")

    def load_network_architecture(self,network_path):
        """Loads the network architecture from the network path"""
        net_architecture = np.load(
                os.path.join(network_path,'net_architecture.npy')).item()
        self.log("Network architecture loaded from file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))
        return net_architecture

    def load_network_parameters(self, file_name, file_path='.'):
        self.saver.restore( self.sess,
                            os.path.join(file_path,file_name+'.nnprm'))
        self.log('Network parameters loaded from file:\n{}'.format(
                            os.path.join(file_path,file_name+'.nnprm')))

    def save_network_parameters(self, file_name, file_path='.'):
        save_path = self.saver.save( self.sess,
                            os.path.join(file_path,file_name+'.nnprm'))
        self.log('Network parameters saved to file:\n{}'.format(save_path))

    def train_epochs(self, annotated_image_set,
            selection_type="Bodies", annotation_type='Bodies',
            n_epochs=100, m_samples=100, report_every=10,
            annotation_border_ratio=None, sample_ratio=None,
            normalize_samples=False, segment_all=False,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None):
        """Trains the network on a training set for a specified number of
            epochs. It loads a random training set from the annotated_image_set
            on every epoch. Random images will be from taken from
            centroids
            annotated_image_set:  Instance of class AnnotatedImageSet holding
                                  the image and annotation data to train on
            selection_type:       'Bodies' or 'Centroids'
            annotation_type:      'Bodies' or 'Centroids'
            n_epochs:             Number of training epochs
            report_every:         Print a report every # of epochs
            m_samples:            number of training samples
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            sample_ratio:      List with ratio of samples per groups (sum=1)
            normalize_samples: Scale each individual channel to its maximum
            segment_all:       Segments all instead of single annotations (T/F)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            """
        t_start = time.time()
        now = datetime.datetime.now()
        self.log("\n-------- Start training network @ {} --------".format(
            now.strftime("%Y-%m-%d %H:%M") ) )
        self.log("n_epochs: {}".format(n_epochs))
        self.log("m_samples: {}".format(m_samples))
        self.log("selection_type: {}".format(selection_type))
        self.log("annotation_type: {}".format(annotation_type))
        self.log("annotation_border_ratio: {}".format(annotation_border_ratio))
        self.log("sample_ratio: {}".format(sample_ratio))
        self.log("normalize_samples: {}".format(normalize_samples))
        self.log("segment_all: {}".format(segment_all))
        self.log("morph_annotations: {}".format(morph_annotations))

        # Loop across training epochs
        for epoch_no in range(n_epochs):

            # Get samples and labels for this epoch
            samples,labels,annotations = annotated_image_set.data_sample(
                zoom_size=(self.y_res,self.x_res),
                annotation_type=selection_type, m_samples=m_samples,
                return_size=(self.y_res_out,self.x_res_out),
                return_annotations=annotation_type,
                annotation_border_ratio=annotation_border_ratio,
                sample_ratio=sample_ratio,
                normalize_samples=normalize_samples, segment_all=segment_all,
                morph_annotations=morph_annotations,
                rotation_list=rotation_list, scale_list_x=scale_list_x,
                scale_list_y=scale_list_y, noise_level_list=noise_level_list )

            # Report progress at start of training
            if (epoch_no % report_every) == 0:
                if epoch_no>0:
                    print( (report_every*'\b')+(report_every*' '),
                        end="", flush=True )
                self.report_progress( samples, annotations,
                    epoch_no, 'Epoch no', t_start)

            # Train the network on samples and labels
            self.sess.run( self.train_step, feed_dict={
                self.x: samples, self.y_trgt: annotations,
                self.fc_keep_prob: self.fc_dropout } )
            print('.', end="", flush=True)

            # Update total number of trained samples
            self.n_samples_trained += m_samples

        self.log("\nNetwork has now been trained on a total of {} samples".format(
                self.n_samples_trained))
        now = datetime.datetime.now()
        self.log("Done @ {}\n".format(
            now.strftime("%Y-%m-%d %H:%M") ) )

    def train_batch(self, annotated_image_set,
            selection_type="Bodies", annotation_type='Bodies',
            n_batches=10, n_epochs=100, batch_size=1000, m_samples=100,
            annotation_border_ratio=None, sample_ratio=None,
            normalize_samples=False, segment_all=False,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None):
        """Trains the network on a training set for a specified number of
            batches of size batch_size. Every batch iteration it loads a
            random training batch from the annotated_image_set. Per batch,
            training is done for n_epochs on a random sample of size m_samples
            that is selected from the current batch.
            annotated_image_set:  Instance of class AnnotatedImageSet holding
                                  the image and annotation data to train on
            selection_type:       'Bodies' or 'Centroids'
            annotation_type:      'Bodies' or 'Centroids'
            n_batches:            Number of batches to run
            n_epochs:             Number of training epochs
            batch_size:           Number of training samples in batch
            m_samples:            Number of training samples in epoch
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            sample_ratio:      List with ratio of samples per groups (sum=1)
            normalize_samples: Scale each individual channel to its maximum
            segment_all:       Segments all instead of single annotations (T/F)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            """

        t_start = time.time()
        now = datetime.datetime.now()
        self.log("\n-------- Start training network @ {} --------".format(
            now.strftime("%Y-%m-%d %H:%M") ) )
        self.log("n_batches: {}".format(n_batches))
        self.log("batch_size: {}".format(batch_size))
        self.log("n_epochs: {}".format(n_epochs))
        self.log("m_samples: {}".format(m_samples))
        self.log("selection_type: {}".format(selection_type))
        self.log("annotation_type: {}".format(annotation_type))
        self.log("annotation_border_ratio: {}".format(annotation_border_ratio))
        self.log("sample_ratio: {}".format(sample_ratio))
        self.log("normalize_samples: {}".format(normalize_samples))
        self.log("segment_all: {}".format(segment_all))
        self.log("morph_annotations: {}".format(morph_annotations))

        # Loop across training batches
        for batch_no in range(n_batches):

            # Get batch of samples and labels
            samples,labels,annotations = annotated_image_set.data_sample(
                zoom_size=(self.y_res,self.x_res),
                annotation_type=selection_type, m_samples=batch_size,
                return_size=(self.y_res_out,self.x_res_out),
                return_annotations=annotation_type,
                annotation_border_ratio=annotation_border_ratio,
                sample_ratio=sample_ratio,
                normalize_samples=normalize_samples, segment_all=segment_all,
                morph_annotations=morph_annotations,
                rotation_list=rotation_list, scale_list_x=scale_list_x,
                scale_list_y=scale_list_y, noise_level_list=noise_level_list )

            # Report progress at start of training
            self.report_progress( samples, annotations,
                batch_no, 'Batch no', t_start)

            # Train the network for n_epochs on random subsets of m_samples
            for epoch_no in range(n_epochs):
                # indices of random samples
                sample_ixs = np.random.choice(
                                batch_size, m_samples, replace=False )
                epoch_samples = samples[ sample_ixs, : ]
                epoch_annotations = annotations[ sample_ixs, : ]
                self.sess.run( self.train_step, feed_dict={
                    self.x: epoch_samples, self.y_trgt: epoch_annotations,
                    self.fc_keep_prob: self.fc_dropout } )
                print('.', end="", flush=True)

            # Update total number of trained samples
            self.n_samples_trained += m_samples

        self.log("\nNetwork has now been trained on a total of {} samples".format(
                self.n_samples_trained))
        now = datetime.datetime.now()
        self.log("Done @ {}\n".format(
            now.strftime("%Y-%m-%d %H:%M") ) )

    def log(self, line_text, no_enter=False, overwrite_last=False):
        """Output to log file and prints to screen"""
        if self.logging:
            with open(os.path.join(self.network_path,'activity.log'), 'a+') as f:
                f.write(line_text + os.linesep)
        if overwrite_last:
            line_text = "\r" + line_text
        if no_enter:
            print(line_text, end="", flush=True)
        else:
            print(line_text)

    def report_progress(self, samples, annotations,
                            epoch_no, counter_name, t_start):
        """Report progress and accuracy on single line for epoch training
            samples:      2d matrix containing training samples
            annotations:  2d matrix containing annotations
            counter:      Number of current epoch / batch
            counter_name: Name to display as counter
            t_start:      'time.time()' time stamp of start training
        """
        # Calculate network accuracy
        result = self.sess.run( [self.network_prediction], feed_dict={
            self.x: samples, self.y_trgt: annotations,
            self.fc_keep_prob: 1.0 })
        pred = result[0]

        # Calculate true/false pos/neg
        true_pos = np.sum( pred[annotations==1]==1 )
        false_pos = np.sum( pred[annotations==0]==1 )
        false_neg = np.sum( pred[annotations==1]==0 )
        true_neg = np.sum( pred[annotations==0]==0 )

        # Calculate accuracy, precision, recall, F1
        accuracy = (true_pos+true_neg) / (annotations.shape[0]*annotations.shape[1])
        precision = np.nan_to_num(true_pos / (true_pos+false_pos))
        recall = np.nan_to_num(true_pos / (true_pos+false_neg))
        F1 = np.nan_to_num(2 * ((precision*recall)/(precision+recall)))
        self.accuracy_list.append(float(accuracy))
        self.precision_list.append(float(precision))
        self.recall_list.append(float(recall))
        self.F1_list.append(float(F1))
        self.n_samples_list.append(int(self.n_samples_trained))

        t_curr = time.time()
        self.log('{} {:4d}: Acc = {:6.4f} (t={})'.format( \
            counter_name, epoch_no, accuracy,
            str(datetime.timedelta(seconds=np.round(t_curr-t_start))) ),
            no_enter=True, overwrite_last=True)

    def report_F1(self, annotated_image_set,
            selection_type="Bodies", annotation_type='Bodies',
            m_samples=100, channel_order=None,
            annotation_border_ratio=None, sample_ratio=None,
            normalize_samples=False, segment_all=False,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None,
            show_figure='Off'):
        """Loads a random training set from the annotated_image_set and
            reports accuracy, precision, recall and F1 score.
            annotated_image_set:  Instance of class AnnotatedImageSet holding
                                  the image and annotation data to train on
            annotation_type:      'Bodies' or 'Centroids'
            selection_type:       'Bodies' or 'Centroids'
            m_samples:            number of test samples
            channel_order:     Tuple indicating which channels are R, G and B
            sample_ratio:      List with ratio of samples per groups (sum=1)
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            normalize_samples: Scale each individual channel to its maximum
            segment_all:       Segments all instead of single annotations (T/F)
            morph_annotations: Randomly morph the annotations
            rotation_list:     List of rotation values to choose from in degrees
            scale_list_x:      List of horizontal scale factors to choose from
            scale_list_y:      List of vertical scale factors to choose from
            noise_level_list:  List of noise levels to choose from
            show_figure:       Show a figure with example samples containing
                                true/false positives/negatives
            """
        # Get m samples and labels from the AnnotatedImageSet
        samples,labels,annotations = annotated_image_set.data_sample(
            zoom_size=(self.y_res,self.x_res),
            annotation_type=selection_type, m_samples=m_samples,
            return_size=(self.y_res_out,self.x_res_out),
            return_annotations=annotation_type,
            annotation_border_ratio=annotation_border_ratio,
            sample_ratio=sample_ratio,
            normalize_samples=normalize_samples, segment_all=segment_all,
            morph_annotations=morph_annotations,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list )

        # Calculate network accuracy
        result = self.sess.run( [self.network_prediction], feed_dict={
            self.x: samples, self.y_trgt: annotations,
            self.fc_keep_prob: 1.0 } )
        pred = result[0]

        # Calculate true/false pos/neg
        true_pos = np.sum( pred[annotations==1]==1 )
        false_pos = np.sum( pred[annotations==0]==1 )
        false_neg = np.sum( pred[annotations==1]==0 )
        true_neg = np.sum( pred[annotations==0]==0 )

        # Calculate accuracy, precision, recall, F1
        final_accuracy = (true_pos+true_neg) / (annotations.shape[0]*annotations.shape[1])
        final_precision = true_pos / (true_pos+false_pos)
        final_recall = true_pos / (true_pos+false_neg)
        final_F1 = \
            2 * ((final_precision*final_recall)/(final_precision+final_recall))
        self.log('Labeled image set of size m={}:'.format(m_samples))
        self.log(' - # true positives = {:6.0f}'.format( true_pos ))
        self.log(' - # false positives = {:6.0f}'.format( false_pos ))
        self.log(' - # false negatives = {:6.0f}'.format( false_neg ))
        self.log(' - # true negatives = {:6.0f}'.format( true_neg ))
        self.log(' - Accuracy = {:6.4f}'.format( final_accuracy ))
        self.log(' - Precision = {:6.4f}'.format( final_precision ))
        self.log(' - Recall = {:6.4f}'.format( final_recall ))
        self.log(' - F1-score = {:6.4f}'.format( final_F1 ))

        # Display figure with examples if necessary
        if show_figure.lower() == 'on':
            plot_positions = [(0,0),(0,1),(1,0),(1,1)]

            sample_no = np.zeros(m_samples)
            true_pos = np.zeros(m_samples)
            false_pos = np.zeros(m_samples)
            false_neg = np.zeros(m_samples)
            true_neg = np.zeros(m_samples)
            for s in range(m_samples):
                s_pred = pred[s,:]
                sample_no[s] = s
                true_pos[s] = np.sum( s_pred[annotations[s,:]==1]==1 )
                false_pos[s] = np.sum( s_pred[annotations[s,:]==0]==1 )
                false_neg[s] = np.sum( s_pred[annotations[s,:]==1]==0 )
                true_neg[s] = np.sum( s_pred[annotations[s,:]==0]==0 )

            accuracy = (true_pos+true_neg) / annotations.shape[1]
            precision = true_pos / (true_pos+false_pos)
            recall = true_pos / (true_pos+false_neg)
            F1 = 2 * ((precision*recall)/(precision+recall))

            all_values = accuracy+precision+recall+F1
            sample_no = sample_no[np.isfinite(all_values)].astype(np.int)
            accuracy = accuracy[np.isfinite(all_values)]
            precision = precision[np.isfinite(all_values)]
            recall = recall[np.isfinite(all_values)]
            F1 = F1[np.isfinite(all_values)]

            n_col = 8
            n_row = 2
            n_show = n_row*n_col
            for worst_best in range(2):
                if worst_best == 0:
                    sort_nr = 1
                    titles = ["worst accuracy","worst precision","worst recall","worst F1"]
                    samples_mat = []
                    annot_mat = []
                    pred_mat = []
                    sample_ix = sample_no[ accuracy.argsort()[:n_show][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                    sample_ix = sample_no[ precision.argsort()[:n_show][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                    sample_ix = sample_no[ recall.argsort()[:n_show][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                    sample_ix = sample_no[ F1.argsort()[:n_show][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                else:
                    sort_nr = -1
                    titles = ["best accuracy","best precision","best recall","best F1"]
                    samples_mat = []
                    annot_mat = []
                    pred_mat = []
                    sample_ix = sample_no[ accuracy.argsort()[(-n_show):][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                    sample_ix = sample_no[ precision.argsort()[(-n_show):][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                    sample_ix = sample_no[ recall.argsort()[(-n_show):][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )
                    sample_ix = sample_no[ F1.argsort()[(-n_show):][::sort_nr] ]
                    samples_mat.append( samples[sample_ix,:] )
                    annot_mat.append( annotations[sample_ix,:] )
                    pred_mat.append( pred[sample_ix,:] )

                # Handle RGB channel order
                if channel_order == None:
                    chan_order = []
                    for ch in range(3):
                        if ch < self.n_input_channels:
                            chan_order.append(ch)
                else:
                    chan_order = channel_order

                plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
                for cnt in range(4):
                    grid_im,_,brdr = ia.image_grid_RGB( samples_mat[cnt],
                        n_channels=annotated_image_set.n_channels,
                        image_size=(self.y_res,self.x_res), n_x=n_col, n_y=n_row,
                        channel_order=chan_order, amplitude_scaling=(1.33,1.33,1),
                        line_color=1, auto_scale=True, return_borders=True )
                    if self.n_input_channels > 2:
                        grid_im[:,:,2] = 0 # only show red and green channel
                        grid_im[brdr==1] = 1 # Make borders white

                    grid_annot,_,brdr = ia.image_grid_RGB( annot_mat[cnt],
                        n_channels=1,
                        image_size=(self.y_res_out,self.x_res_out), n_x=n_col, n_y=n_row,
                        amplitude_scaling=(1.33,1.33,1),
                        line_color=1, auto_scale=True, return_borders=True )
                    grid_pred,_ = ia.image_grid_RGB( pred_mat[cnt],
                        n_channels=1,
                        image_size=(self.y_res_out,self.x_res_out), n_x=n_col, n_y=n_row,
                        amplitude_scaling=(1.33,1.33,1),
                        line_color=1, auto_scale=True )
                    grid_annot[:,:,1] = 0 # Annotations in red, prediction in blue
                    grid_annot[:,:,2] = grid_pred[:,:,0]
                    grid_annot[brdr==1] = 1 # Make borders white

                    with sns.axes_style("white"):
                        ax = plt.subplot2grid( (4,2), (cnt,0) )
                        ax.imshow( grid_im,
                            interpolation='nearest', vmax=grid_im.max()*0.8 )
                        ax.set_title("Input images: " + titles[cnt])
                        plt.axis('tight')
                        plt.axis('off')

                        ax = plt.subplot2grid( (4,2), (cnt,1) )
                        ax.imshow( grid_annot,
                            interpolation='nearest', vmax=grid_annot.max() )
                        ax.set_title("Annotations: " + titles[cnt])
                        plt.axis('tight')
                        plt.axis('off')
                plt.tight_layout()

    def show_learning_curve(self):
        """Displays a learning curve of accuracy versus number of
        trained samples"""

        # Get data
        x_values = np.array(self.n_samples_list)
        accuracy = np.array(self.accuracy_list)
        precision = np.array(self.precision_list)
        recall = np.array(self.recall_list)
        F1 = np.array(self.F1_list)

        # Make plot
        with sns.axes_style("ticks"):
            fig,ax = plt.subplots()
            plt.plot([np.min(x_values),np.max(x_values)],[0.5,0.5],
                        color='#777777',linestyle='--')
            plt.plot([np.min(x_values),np.max(x_values)],[0.66,0.66],
                        color='#777777',linestyle=':')
            plt.plot([np.min(x_values),np.max(x_values)],[0.8,0.8],
                        color='#777777',linestyle=':')
            plt.plot([np.min(x_values),np.max(x_values)],[0.9,0.9],
                        color='#777777',linestyle=':')

            plt.plot( x_values, accuracy, color='#000000',
                        linewidth=1, label='Accuracy' )
            plt.plot( x_values, precision, color='#0000aa',
                        linewidth=1, label='Precision' )
            plt.plot( x_values, recall, color='#00aa00',
                        linewidth=1, label='Recall' )
            plt.plot( x_values, F1, color='#aa0000',
                        linewidth=2, label='F1' )

            plt.yticks( [0, 0.5, 0.66, 0.8, 0.9, 1.0],
                ['0','0.5','0.66','0.8','0.9','1.0'], ha='right' )
            plt.xlim(np.max(x_values)*-0.02,np.max(x_values)*1.02)
            plt.ylim(-0.02,1.02)
            plt.xlabel('Number of training samples')
            plt.ylabel('Performance')
            plt.title('Learning curve')
            sns.despine(ax=ax, offset=0, trim=True)
            lgnd = plt.legend(loc=4, ncol=1, frameon=True, fontsize=9)
            lgnd.get_frame().set_facecolor('#ffffff')
            ax.spines['left'].set_bounds(0,1)
            ax.spines['bottom'].set_bounds(np.min(x_values),np.max(x_values))


########################################################################
### Deep convolutional neural network
### 2 conv layers, one dense layer, n output units
########################################################################

class ConvNetCnvNFc1Nout(NeuralNetSegmentation):
    """Holds a deep convolutional neural network for annotating images.
    N convolutional layers, 1 fully connected layer, 1 output layer"""

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                output_image_size=None,
                conv_n_layers=3, conv_size=5, conv_n_chan=32, conv_n_pool=2,
                fc1_n_chan=1024, fc_dropout=0.5, alpha=4e-4 ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing (y,x) size of the network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):

            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new ConvNetCnv2Fc1Nout network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.y_res_out = output_image_size[0]
            self.x_res_out = output_image_size[1]
            self.n_output_pixels = output_image_size[0]*output_image_size[1]
            self.conv_n_layers = conv_n_layers
            self.conv_size = conv_size
            self.conv_n_chan = conv_n_chan
            self.conv_n_pool = conv_n_pool

            self.fc1_y_size = self.y_res
            self.fc1_x_size = self.x_res
            self.conv_n_chan_L = [ self.n_input_channels ]
            for L in range(self.conv_n_layers):
                self.conv_n_chan_L.append(
                    int(self.conv_n_chan * (self.conv_n_pool**L)) )
                self.fc1_y_size = np.ceil(self.fc1_y_size/self.conv_n_pool)
                self.fc1_x_size = np.ceil(self.fc1_x_size/self.conv_n_pool)
            self.fc1_y_size = int(self.fc1_y_size)
            self.fc1_x_size = int(self.fc1_x_size)

            self.fc1_n_chan = fc1_n_chan
            self.fc_dropout = fc_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_samples_list = []
            self.accuracy_list = []
            self.precision_list = []
            self.recall_list = []
            self.F1_list = []

            # Save network architecture
            self.save_network_architecture( network_path=self.network_path )

        else:
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Re-initialization of existing network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    " ")

            # Load network architecture from directory
            net_architecture = self.load_network_architecture(self.network_path)

            # Set up network variables from loaded architecture
            self.y_res = net_architecture['y_res']
            self.x_res = net_architecture['x_res']
            self.n_input_channels = net_architecture['n_input_channels']
            self.y_res_out = net_architecture['y_res_out']
            self.x_res_out = net_architecture['x_res_out']
            self.n_output_pixels = net_architecture['n_output_pixels']

            self.conv_n_layers = conv_n_layers
            self.conv_size = conv_size
            self.conv_n_chan = conv_n_chan
            self.conv_n_pool = conv_n_pool

            self.fc1_y_size = self.y_res
            self.fc1_x_size = self.x_res
            self.conv_n_chan_L = [ self.n_input_channels ]
            for L in range(self.conv_n_layers):
                self.conv_n_chan_L.append(
                    int(self.conv_n_chan * (self.conv_n_pool**L)) )
                self.fc1_y_size = np.ceil(self.fc1_y_size/self.conv_n_pool)
                self.fc1_x_size = np.ceil(self.fc1_x_size/self.conv_n_pool)
            self.fc1_y_size = int(self.fc1_y_size)
            self.fc1_x_size = int(self.fc1_x_size)

            self.fc1_n_chan = net_architecture['fc1_n_chan']
            self.fc_dropout = net_architecture['fc_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc_dropout != fc_dropout:
            self.fc_dropout = fc_dropout
            self.log("Updated dropout fraction to {}".format(self.fc_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                                        shape = [None, self.n_output_pixels] )

        # Convert input image to tensor with channel as last dimension
        # x_image = [-1 x im-height x im-width x n-input-channels]
        x_image_temp = tf.reshape(self.x, [-1,
            self.n_input_channels,self.y_res,self.x_res])
        x_image = tf.transpose(x_image_temp, [0,2,3,1])

        #########################################################
        # Set up convolutional layers
        self.conv_shape = []
        self.W_conv = []
        self.b_conv = []
        self.conv_lin = []
        self.conv_relu = []
        self.conv_kernel = []
        self.conv_pool = []

        # Loop layers
        for L in range(self.conv_n_layers):

            # W = [im-height x im-width x n-input-channels x n-output-channels])
            self.conv_shape.append( [self.conv_size, self.conv_size,
                self.conv_n_chan_L[L], self.conv_n_chan_L[L+1]] )
            self.W_conv.append( tf.Variable( tf.truncated_normal(
                                    shape=self.conv_shape[L], stddev=0.1)) )
            self.b_conv.append( tf.Variable( tf.constant(0.1,
                                    shape=[self.conv_n_chan_L[L+1]] )) )

            # Convolve x_image with the weight tensor
            if L == 0:
                self.conv_lin.append( tf.nn.conv2d( x_image,
                    self.W_conv[L], strides=[1, 1, 1, 1], padding='SAME' ) )
            else:
                self.conv_lin.append( tf.nn.conv2d( self.conv_pool[L-1],
                    self.W_conv[L], strides=[1, 1, 1, 1], padding='SAME' ) )

            # Add bias and apply transfer function
            self.conv_relu.append(
                tf.nn.relu( self.conv_lin[L] + self.b_conv[L] ) )

            # Max pooling
            self.conv_kernel.append([1, self.conv_n_pool, self.conv_n_pool, 1])
            self.conv_pool.append( tf.nn.max_pool(
                self.conv_relu[L], ksize=self.conv_kernel[L],
                strides=self.conv_kernel[L], padding='SAME') )

        #########################################################
        # Densely Connected Layer
        # Weights and bias
        self.fc1_shape = [self.fc1_y_size * self.fc1_x_size * self.conv_n_chan_L[-1],
                          self.fc1_n_chan]
        self.W_fc1 = tf.Variable( tf.truncated_normal(
                               shape=self.fc1_shape, stddev=0.1 ) )
        self.b_fc1 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Flatten output from conv2
        self.conv_last_pool_flat = tf.reshape(
            self.conv_pool[-1], [-1, self.fc1_shape[0]] )

        # Calculate network step
        self.fc1_relu = tf.nn.relu( tf.matmul( self.conv_last_pool_flat,
            self.W_fc1) + self.b_fc1 )

        # Set up dropout option for fc1
        self.fc1_keep_prob = tf.placeholder(tf.float32)
        self.fc1_relu_drop = tf.nn.dropout(self.fc1_relu, self.fc1_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = [self.fc1_n_chan, self.n_output_pixels]
        self.W_fc_out = tf.Variable( tf.truncated_normal(
                                shape=self.fc_out_shape, stddev=0.1 ) )
        self.b_fc_out = tf.Variable( tf.constant(0.1,
                                shape=[self.fc_out_shape[1]] ))

        # Calculate network step
        self.fc_out_lin = tf.matmul( self.fc1_relu_drop,
                                     self.W_fc_out ) + self.b_fc_out

        #########################################################
        # Define cost function and optimizer algorithm
        self.cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.threshold = tf.constant( 0.5, dtype=tf.float32 )
        self.network_prediction  = tf.cast( tf.greater(
                                        self.fc_out_lin, self.threshold ), tf.float32 )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("y_res_out: {}".format(self.y_res_out))
        self.log("x_res_out: {}".format(self.x_res_out))
        self.log("n_output_pixels: {}".format(self.n_output_pixels))
        self.log("conv_n_layers: {}".format(self.conv_n_layers))
        for L in range(self.conv_n_layers):
            self.log("conv{}_size: {}".format(L+1,self.conv_size))
            self.log("conv{}_n_input_chan: {}".format(L+1,self.conv_n_chan_L[L]))
            self.log("conv{}_n_output_chan: {}".format(L+1,self.conv_n_chan_L[L+1]))
            self.log("conv{}_n_pool: {}".format(L+1,self.conv_n_pool))
        self.log("fc1_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc_dropout: {}".format(self.fc_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['y_res_out'] = self.y_res_out
        net_architecture['x_res_out'] = self.x_res_out
        net_architecture['n_output_pixels'] = self.n_output_pixels
        net_architecture['conv_n_layers'] = self.conv_n_layers
        net_architecture['conv_size'] = self.conv_size
        net_architecture['conv_n_chan'] = self.conv_n_chan
        net_architecture['conv_n_pool'] = self.conv_n_pool
        net_architecture['fc1_n_chan'] = self.fc1_n_chan
        net_architecture['fc_dropout'] = self.fc_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['accuracy_list'] = self.accuracy_list
        net_architecture['precision_list'] = self.precision_list
        net_architecture['recall_list'] = self.recall_list
        net_architecture['F1_list'] = self.F1_list
        np.save(os.path.join( \
            network_path,'net_architecture.npy'), net_architecture)
        self.log("Network architecture saved to file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))

    def show_filters(self):
        """Plot the convolutional filters"""

########################################################################
### Deep convolutional neural network
### 1 conv layer, two densely connected layers, n output units
########################################################################

class ConvNetCnv1Fc2Nout(NeuralNetSegmentation):
    """Holds a deep convolutional neural network for annotating images.
    1 convolutional layer, 2 fully connected layers, 1 output layer"""

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                output_image_size=None,
                conv1_size=5, conv1_n_chan=32, conv1_n_pool=2,
                fc1_n_chan=1024, fc2_n_chan=1024,
                fc_dropout=0.5, alpha=4e-4 ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing (y,x) size of the network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):

            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new ConvNetCnv2Fc1Nout network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.y_res_out = output_image_size[0]
            self.x_res_out = output_image_size[1]
            self.n_output_pixels = output_image_size[0]*output_image_size[1]
            self.conv1_size = conv1_size
            self.conv1_n_chan = conv1_n_chan
            self.conv1_n_pool = conv1_n_pool
            self.fc1_y_size = int( np.ceil( self.y_res/self.conv1_n_pool ) )
            self.fc1_x_size = int( np.ceil( self.x_res/self.conv1_n_pool ) )
            self.fc1_n_chan = fc1_n_chan
            self.fc2_n_chan = fc2_n_chan
            self.fc_dropout = fc_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_samples_list = []
            self.accuracy_list = []
            self.precision_list = []
            self.recall_list = []
            self.F1_list = []

            # Save network architecture
            self.save_network_architecture( network_path=self.network_path )

        else:
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Re-initialization of existing network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    " ")

            # Load network architecture from directory
            net_architecture = self.load_network_architecture(self.network_path)

            # Set up network variables from loaded architecture
            self.y_res = net_architecture['y_res']
            self.x_res = net_architecture['x_res']
            self.n_input_channels = net_architecture['n_input_channels']
            self.y_res_out = net_architecture['y_res_out']
            self.x_res_out = net_architecture['x_res_out']
            self.n_output_pixels = net_architecture['n_output_pixels']
            self.conv1_size = net_architecture['conv1_size']
            self.conv1_n_chan = net_architecture['conv1_n_chan']
            self.conv1_n_pool = net_architecture['conv1_n_pool']
            self.fc1_y_size = int( np.ceil( self.y_res/self.conv1_n_pool ) )
            self.fc1_x_size = int( np.ceil( self.x_res/self.conv1_n_pool ) )
            self.fc1_n_chan = net_architecture['fc1_n_chan']
            self.fc2_n_chan = net_architecture['fc2_n_chan']
            self.fc_dropout = net_architecture['fc_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc_dropout != fc_dropout:
            self.fc_dropout = fc_dropout
            self.log("Updated dropout fraction to {}".format(self.fc_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                                        shape = [None, self.n_output_pixels] )

        # Convert input image to tensor with channel as last dimension
        # x_image = [-1 x im-height x im-width x n-input-channels]
        x_image_temp = tf.reshape(self.x, [-1,
            self.n_input_channels,self.y_res,self.x_res])
        x_image = tf.transpose(x_image_temp, [0,2,3,1])

        #########################################################
        # Set up convolutional layer 1
        # W = [im-height x im-width x n-input-channels x n-output-channels])
        self.conv1_shape = [self.conv1_size, self.conv1_size,
                       self.n_input_channels, self.conv1_n_chan]
        self.W_conv1 = tf.Variable( tf.truncated_normal(
                               shape=self.conv1_shape, stddev=0.1))
        self.b_conv1 = tf.Variable( tf.constant(0.1,
                                                shape=[self.conv1_n_chan] ))

        # Convolve x_image with the weight tensor
        self.conv1_lin = tf.nn.conv2d( x_image, self.W_conv1,
                                  strides=[1, 1, 1, 1], padding='SAME' )

        # Add bias and apply transfer function
        self.conv1_relu = tf.nn.relu( self.conv1_lin + self.b_conv1 )

        # Max pooling
        self.conv1_kernel = [1, self.conv1_n_pool, self.conv1_n_pool, 1]
        self.conv1_pool = tf.nn.max_pool( self.conv1_relu,
            ksize=self.conv1_kernel, strides=self.conv1_kernel, padding='SAME')

        #########################################################
        # First densely Connected Layer
        # Weights and bias
        self.fc1_shape = [self.fc1_y_size * self.fc1_x_size * self.conv1_n_chan,
                          self.fc1_n_chan]
        self.W_fc1 = tf.Variable( tf.truncated_normal(
                               shape=self.fc1_shape, stddev=0.1 ) )
        self.b_fc1 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Flatten output from conv2
        self.conv1_pool_flat = tf.reshape(
            self.conv1_pool, [-1, self.fc1_shape[0]] )

        # Calculate network step
        self.fc1_relu = tf.nn.relu( tf.matmul( self.conv1_pool_flat,
            self.W_fc1) + self.b_fc1 )

        #########################################################
        # Second densely Connected Layer
        # Weights and bias
        self.fc2_shape = [self.fc1_n_chan,self.fc2_n_chan]
        self.W_fc2 = tf.Variable( tf.truncated_normal(
                               shape=self.fc2_shape, stddev=0.1 ) )
        self.b_fc2 = tf.Variable( tf.constant(0.1, shape=[self.fc2_n_chan] ))

        # Calculate network step
        self.fc2_relu = tf.nn.relu( tf.matmul( self.fc1_relu,
            self.W_fc2) + self.b_fc2 )

        # Set up dropout option for fc1
        self.fc_keep_prob = tf.placeholder(tf.float32)
        self.fc2_relu_drop = tf.nn.dropout(self.fc2_relu, self.fc_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = [self.fc2_n_chan, self.n_output_pixels]
        self.W_fc_out = tf.Variable( tf.truncated_normal(
                                shape=self.fc_out_shape, stddev=0.1 ) )
        self.b_fc_out = tf.Variable( tf.constant(0.1,
                                shape=[self.fc_out_shape[1]] ))

        # Calculate network step
        self.fc_out_lin = tf.matmul( self.fc2_relu_drop,
                                     self.W_fc_out ) + self.b_fc_out

        #########################################################
        # Define cost function and optimizer algorithm
        self.cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.threshold = tf.constant( 0.5, dtype=tf.float32 )
        self.network_prediction  = tf.cast( tf.greater(
                                        self.fc_out_lin, self.threshold ), tf.float32 )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("y_res_out: {}".format(self.y_res_out))
        self.log("x_res_out: {}".format(self.x_res_out))
        self.log("n_output_pixels: {}".format(self.n_output_pixels))
        self.log("conv1_size: {}".format(self.conv1_size))
        self.log("conv1_n_chan: {}".format(self.conv1_n_chan))
        self.log("conv1_n_pool: {}".format(self.conv1_n_pool))
        self.log("fc1_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc2_n_chan: {}".format(self.fc2_n_chan))
        self.log("fc_dropout: {}".format(self.fc_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['y_res_out'] = self.y_res_out
        net_architecture['x_res_out'] = self.x_res_out
        net_architecture['n_output_pixels'] = self.n_output_pixels
        net_architecture['conv1_size'] = self.conv1_size
        net_architecture['conv1_n_chan'] = self.conv1_n_chan
        net_architecture['conv1_n_pool'] = self.conv1_n_pool
        net_architecture['fc1_n_chan'] = self.fc1_n_chan
        net_architecture['fc2_n_chan'] = self.fc2_n_chan
        net_architecture['fc_dropout'] = self.fc_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['accuracy_list'] = self.accuracy_list
        net_architecture['precision_list'] = self.precision_list
        net_architecture['recall_list'] = self.recall_list
        net_architecture['F1_list'] = self.F1_list
        np.save(os.path.join( \
            network_path,'net_architecture.npy'), net_architecture)
        self.log("Network architecture saved to file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))

    def show_filters(self):
        """Plot the convolutional filters"""

        filter_no = 4
        n_iterations = 40

        # Isolate activation of a single convolutional filter
        layer_slice_begin = tf.constant( [0,0,0,filter_no], dtype=tf.int32 )
        layer_slice_size = tf.constant( [-1,-1,-1,1], dtype=tf.int32 )
        layer_units = tf.slice( self.conv1_relu,
                                layer_slice_begin, layer_slice_size )

        # Define cost function for filter
        layer_activation = tf.reduce_mean( layer_units )

        # Optimize input image for maximizing filter output
        gradients = tf.gradients( ys=layer_activation, xs=[self.x] )[0]

        norm_grad = tf.nn.l2_normalize( gradients, dim=0, epsilon=1e-8)

        # Random staring point
        im = np.random.random( (1,
            self.n_input_channels * self.y_res * self.x_res) )

        # Do a gradient ascend
        return_im = np.zeros( (n_iterations,
            self.n_input_channels * self.y_res * self.x_res) )

        return_im[0,:] = im
        for e in range(n_iterations-1):
            result = self.sess.run( [norm_grad], feed_dict={ self.x: im } )
            # Gradient ASCENT
            im += result[0]
            return_im[e+1,:] = im

        plt.figure(figsize=(12,8), facecolor='w', edgecolor='w')
        for ch in range(self.n_input_channels):

            grid_im,_,brdr = ia.image_grid_RGB( return_im,
                n_channels=self.n_input_channels,
                image_size=(self.y_res,self.x_res), n_x=10, n_y=4,
                channel_order=(ch,ch,ch), amplitude_scaling=(1.33,1.33,1),
                line_color=1, auto_scale=True, return_borders=True )

            ax = plt.subplot2grid( (self.n_input_channels,1), (ch,0) )
            with sns.axes_style("white"):
                ax.imshow( grid_im,
                    interpolation='nearest', vmax=grid_im.max() )
                ax.set_title("Filter {}, ch {}".format(filter_no,ch))
                plt.axis('tight')
                plt.axis('off')
        plt.tight_layout()
