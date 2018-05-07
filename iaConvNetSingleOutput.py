#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 8 12:15:25 2017

Contains functions that set up a convolutional neural net for image annotation

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

# tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################################################################
### Single output neural network class
########################################################################

class NeuralNetSingleOutput(object):
    """Base class holding a neural network for annotating (multi channel)
    images with single pixel output (but multi class clasification)
    See NeuralNetBase for more info"""

    def __init__(self):
        """Does nothing, initialization should be done by subclasses
        List of required subclass variables:

        self.logging:               (True/False)
        self.network_path:          Path
        self.y_res:                 y resolution of input image patches
        self.x_res:                 x resolution of input image patches
        self.n_input_channels:      Number of input channels
        self.n_output_classes:      Number of output classes
        self.fc1_dropout:           Dropout fraction of network during training
        self.alpha:                 Learning rate

        self.x:                     Place holder for input data
        self.y_trgt:                Place holder for training target
        self.fc_out_lin:            Output layer
        self.fc1_keep_prob:         Place holder for dropout value
        self.train_step:            Optimizer OP
        self.network_prediction:    Prediction of network
        self.accuracy:              Prediction accuracy of network
        self.saver:                 Saver OP

        self.n_samples_trained
        self.n_class_samples_trained
        self.n_samples_list
        self.n_class_samples_list
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
        self.log("Network architecture loaded from file: {}".format(
                            os.path.join(network_path,'net_architecture.npy')))
        return net_architecture

    def load_network_parameters(self, file_name, file_path='.'):
        self.saver.restore( self.sess,
                            os.path.join(file_path,file_name+'.nnprm'))
        self.log('Network parameters loaded from file: {}'.format(
                            os.path.join(file_path,file_name+'.nnprm')))

    def save_network_parameters(self, file_name, file_path='.'):
        save_path = self.saver.save( self.sess,
                            os.path.join(file_path,file_name+'.nnprm'))
        self.log('Network parameters saved to file:\n{}'.format(save_path))

    def train_epochs(self, annotated_image_set, n_epochs=100, report_every=10,
            annotation_type='Bodies', m_samples=100,
            sample_ratio=None, normalize_samples=False,
            annotation_border_ratio=None,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None, noise_level_list=None):
        """Trains the network on a training set for a specified number of
            epochs. It loads a random training set from the annotated_image_set
            on every epoch
            annotated_image_set:  Instance of class AnnotatedImageSet holding
                                  the image and annotation data to train on
            n_epochs:             Number of training epochs
            report_every:         Print a report every # of epochs
            annotation_type:      'Bodies' or 'Centroids'
            m_samples:            number of training samples
            sample_ratio:      List with ratio of samples per groups (sum=1)
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            normalize_samples: Scale each individual channel to its maximum
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
        self.log("annotation_type: {}".format(annotation_type))
        self.log("sample_ratio: {}".format(sample_ratio))
        self.log("annotation_border_ratio: {}".format(annotation_border_ratio))
        self.log("normalize_samples: {}".format(normalize_samples))
        self.log("morph_annotations: {}".format(morph_annotations))

        # Loop across training epochs
        for epoch_no in range(n_epochs):

            # Get samples and labels for this epoch
            samples,labels,annotations = annotated_image_set.data_sample(
                zoom_size=(self.y_res,self.x_res),
                annotation_type=annotation_type, m_samples=m_samples,
                return_annotations=False, sample_ratio=sample_ratio,
                annotation_border_ratio=annotation_border_ratio,
                normalize_samples=normalize_samples,
                morph_annotations=morph_annotations,
                rotation_list=rotation_list, scale_list_x=scale_list_x,
                scale_list_y=scale_list_y, noise_level_list=noise_level_list )

            # Report progress at start of training
            if (epoch_no % report_every) == 0:
                if epoch_no>0:
                    print( (report_every*'\b')+(report_every*' '),
                        end="", flush=True )
                self.report_progress( samples, labels,
                    epoch_no, 'Epoch no', t_start)

            # Train the network on samples and labels
            self.sess.run( self.train_step, feed_dict={
                self.x: samples, self.y_trgt: labels,
                self.fc1_keep_prob: self.fc1_dropout } )
            print('.', end="", flush=True)

            # Update total number of trained samples
            self.n_samples_trained += m_samples
            for c in range(self.n_output_classes):
                self.n_class_samples_trained[c] += int(labels[:,c].sum())

        self.log("\nNetwork has now been trained on a total of {} samples".format(
                self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )
        now = datetime.datetime.now()
        self.log("Done @ {}\n".format(
            now.strftime("%Y-%m-%d %H:%M") ) )

    def train_batch(self, annotated_image_set, n_batches=10, n_epochs=100,
            annotation_type='Bodies', batch_size=1000, m_samples=100,
            sample_ratio=None, annotation_border_ratio=None,
            normalize_samples=False, morph_annotations=False,
            rotation_list=None, scale_list_x=None,
            scale_list_y=None, noise_level_list=None):
        """Trains the network on a training set for a specified number of
            batches of size batch_size. Every batch iteration it loads a
            random training batch from the annotated_image_set. Per batch,
            training is done for n_epochs on a random sample of size m_samples
            that is selected from the current batch.
            annotated_image_set:  Instance of class AnnotatedImageSet holding
                                  the image and annotation data to train on
            n_batches:            Number of batches to run
            n_epochs:             Number of training epochs
            annotation_type:      'Bodies' or 'Centroids'
            batch_size:           Number of training samples in batch
            m_samples:            Number of training samples in epoch
            sample_ratio:      List with ratio of samples per groups (sum=1)
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            normalize_samples: Scale each individual channel to its maximum
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
        self.log("annotation_type: {}".format(annotation_type))
        self.log("sample_ratio: {}".format(sample_ratio))
        self.log("annotation_border_ratio: {}".format(annotation_border_ratio))
        self.log("normalize_samples: {}".format(normalize_samples))
        self.log("morph_annotations: {}".format(morph_annotations))

        # Loop across training batches
        for batch_no in range(n_batches):

            # Get batch of samples and labels
            samples,labels,annotations = annotated_image_set.data_sample(
                zoom_size=(self.y_res,self.x_res),
                annotation_type=annotation_type, m_samples=batch_size,
                return_annotations=False,  sample_ratio=sample_ratio,
                annotation_border_ratio=annotation_border_ratio,
                normalize_samples=normalize_samples,
                morph_annotations=morph_annotations,
                rotation_list=rotation_list, scale_list_x=scale_list_x,
                scale_list_y=scale_list_y, noise_level_list=noise_level_list )

            # Report progress at start of training
            self.report_progress( samples, labels,
                batch_no, 'Batch no', t_start)

            # Train the network for n_epochs on random subsets of m_samples
            for epoch_no in range(n_epochs):
                # indices of random samples
                sample_ixs = np.random.choice(
                                batch_size, m_samples, replace=False )
                epoch_samples = samples[ sample_ixs, : ]
                epoch_labels = labels[ sample_ixs, : ]
                self.sess.run( self.train_step, feed_dict={
                    self.x: epoch_samples, self.y_trgt: epoch_labels,
                    self.fc1_keep_prob: self.fc1_dropout } )
                print('.', end="", flush=True)

            # Update total number of trained samples
            self.n_samples_trained += m_samples
            for c in range(self.n_output_classes):
                self.n_class_samples_trained[c] += int(labels[:,c].sum())

        self.log("\nNetwork has now been trained on a total of {} samples".format(
                self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )
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

    def report_progress(self, samples, labels,
                            epoch_no, counter_name, t_start):
        """Report progress and accuracy on single line for epoch training
            samples:      2d matrix containing training samples
            labels:       2d matrix containing labels
            counter:      Number of current epoch / batch
            counter_name: Name to display as counter
            t_start:      'time.time()' time stamp of start training
        """
        # Calculate network accuracy
        result = self.sess.run( [self.network_prediction], feed_dict={
            self.x: samples, self.y_trgt: labels,
            self.fc1_keep_prob: 1.0 })
        pred = result[0]

        # Loop for each output class
        avg_accuracy = 0
        for c in range(self.n_output_classes):
            # Calculate true/false pos/neg
            true_pos = np.sum( pred[labels[:,c]==1]==c )
            false_pos = np.sum( pred[labels[:,c]==0]==c )
            false_neg = np.sum( pred[labels[:,c]==1]!=c )
            true_neg = np.sum( pred[labels[:,c]==0]!=c )

            # Calculate accuracy, precision, recall, F1
            accuracy = (true_pos+true_neg) / len(pred)
            precision = np.nan_to_num(true_pos / (true_pos+false_pos))
            recall = np.nan_to_num(true_pos / (true_pos+false_neg))
            F1 = np.nan_to_num(2 * ((precision*recall)/(precision+recall)))
            self.accuracy_list[c].append(float(accuracy))
            self.precision_list[c].append(float(precision))
            self.recall_list[c].append(float(recall))
            self.F1_list[c].append(float(F1))
            self.n_class_samples_list[c].append(int(self.n_class_samples_trained[c]))
            avg_accuracy += accuracy
        self.n_samples_list.append(int(self.n_samples_trained))

        t_curr = time.time()
        self.log('{} {:4d}: Acc = {:6.4f} (t={})'.format( \
            counter_name, epoch_no, avg_accuracy/self.n_output_classes,
            str(datetime.timedelta(seconds=np.round(t_curr-t_start))) ),
            no_enter=True, overwrite_last=True)

    def report_F1(self, annotated_image_set, annotation_type='Bodies',
            m_samples=100,  sample_ratio=None,
            annotation_border_ratio=None,
            channel_order=None,  normalize_samples=False,
            morph_annotations=False, rotation_list=None,
            scale_list_x=None, scale_list_y=None,
            noise_level_list=None, show_figure='Off'):
        """Loads a random training set from the annotated_image_set and
            reports accuracy, precision, recall and F1 score.
            annotated_image_set:  Instance of class AnnotatedImageSet holding
                                  the image and annotation data to train on
            annotation_type:      'Bodies' or 'Centroids'
            m_samples:            number of test samples
            sample_ratio:      List with ratio of samples per groups (sum=1)
            annotation_border_ratio: Fraction of samples drawn from 2px border
                               betweem positive and negative samples
            channel_order:     Tuple indicating which channels are R, G and B
            normalize_samples: Scale each individual channel to its maximum
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
            zoom_size=(self.y_res,self.x_res), annotation_type=annotation_type,
            m_samples=m_samples,
            return_annotations=False,  sample_ratio=sample_ratio,
            annotation_border_ratio=annotation_border_ratio,
            normalize_samples=normalize_samples,
            morph_annotations=morph_annotations,
            rotation_list=rotation_list, scale_list_x=scale_list_x,
            scale_list_y=scale_list_y, noise_level_list=noise_level_list )

        # Calculate network accuracy
        result = self.sess.run( [self.network_prediction], feed_dict={
            self.x: samples, self.y_trgt: labels,
            self.fc1_keep_prob: 1.0 })
        pred = result[0]

        # Loop output classes
        for c in range(1,self.n_output_classes):
            # Calculate true/false pos/neg
            true_pos = np.sum( pred[labels[:,c]==1]==c )
            false_pos = np.sum( pred[labels[:,c]==0]==c )
            false_neg = np.sum( pred[labels[:,c]==1]!=c )
            true_neg = np.sum( pred[labels[:,c]==0]!=c )

            # Calculate accuracy, precision, recall, F1
            final_accuracy = (true_pos+true_neg) / len(pred)
            final_precision = true_pos / (true_pos+false_pos)
            final_recall = true_pos / (true_pos+false_neg)
            final_F1 = \
                2 * ((final_precision*final_recall)/(final_precision+final_recall))
            self.log('Labeled image set of size m={}, class {} :'.format(m_samples,c))
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
                titles = ["true positives","false positives",\
                            "false negatives","true negatives"]
                plot_positions = [(0,0),(0,1),(1,0),(1,1)]
                samples_mat = []
                samples_mat.append(
                    samples[ np.logical_and(pred[:]==c,labels[:,c]==1), : ])
                samples_mat.append(
                    samples[ np.logical_and(pred[:]==c,labels[:,c]==0), : ])
                samples_mat.append(
                    samples[ np.logical_and(pred[:]!=c,labels[:,c]==1), : ])
                samples_mat.append(
                    samples[ np.logical_and(pred[:]!=c,labels[:,c]==0), : ])

                # Handle RGB channel order
                if channel_order == None:
                    chan_order = []
                    for ch in range(3):
                        if ch < self.n_input_channels:
                            chan_order.append(ch)
                else:
                    chan_order = channel_order

                plt.figure(figsize=(10,10), facecolor='w', edgecolor='w')
                for cnt in range(4):
                    grid,_,brdr = ia.image_grid_RGB( samples_mat[cnt],
                        n_channels=annotated_image_set.n_channels,
                        image_size=(self.y_res,self.x_res), n_x=10, n_y=10,
                        channel_order=chan_order, amplitude_scaling=(1.33,1.33,1),
                        line_color=1, auto_scale=True, return_borders=True )
                    if self.n_input_channels > 2:
                        grid[:,:,2] = 0 # only show red and green channel
                        grid[brdr==1] = 1 # make borders white
                    with sns.axes_style("white"):
                        ax1 = plt.subplot2grid( (2,2), plot_positions[cnt] )
                        ax1.imshow(
                            grid, interpolation='nearest', vmax=grid.max()*0.8 )
                        ax1.set_title(titles[cnt]+", class={}".format(c))
                        plt.axis('tight')
                        plt.axis('off')
                plt.tight_layout()

    def show_learning_curve(self):
        """Displays a learning curve of accuracy versus number of
        trained samples"""

        # Loop output classes
        for c in range(1,self.n_output_classes):
            # Get data
            x_values = np.array(self.n_class_samples_list[c])
            accuracy = np.array(self.accuracy_list[c])
            precision = np.array(self.precision_list[c])
            recall = np.array(self.recall_list[c])
            F1 = np.array(self.F1_list[c])

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
                plt.title('Learning curve, class {}'.format(c))
                sns.despine(ax=ax, offset=0, trim=True)
                lgnd = plt.legend(loc=4, ncol=1, frameon=True, fontsize=9)
                lgnd.get_frame().set_facecolor('#ffffff')
                ax.spines['left'].set_bounds(0,1)
                ax.spines['bottom'].set_bounds(np.min(x_values),np.max(x_values))

    def annotate_image( self, anim ):
        """Loops through every pixels of an annotated image, classifies
            the pixels and overwrites the annotation list with newly
            detected annotations
            anim:   AnnotatedImage with image channel loaded
            returns a 2d matrix with the classification result
        """
        # Make output matrix
        classified_image = np.zeros((anim.y_res,anim.x_res))

        # Annotate line by line
        line_samples = np.zeros( (anim.x_res,
            anim.n_channels * self.y_res * self.x_res) )
        # Loop through all lines
        print("Annotating image {:6.2f}%".format(0), end="", flush=True)
        for y in range(anim.y_res):
            # Loop through all pixels to fill the line-samples
            for x in range(anim.x_res):
                line_samples[x,:] = ia.image2vec( ia.zoom( anim.channel,
                    y, x, zoom_size=(self.y_res,self.x_res) ) )

            # Calculate network prediction
            result = self.sess.run( [self.network_prediction], feed_dict={
                self.x: line_samples, self.fc1_keep_prob: 1.0 })
            classified_image[y,:] = result[0]
            print((7*'\b')+'{:6.2f}%'.format(100.0*float(y)/float(anim.y_res)),
                        end='', flush=True)
        print( (7*'\b')+'{:6.2f}% .. done!'.format(100.0) )
        return classified_image


########################################################################
### Single layer neural network
########################################################################

class NeuralNet1Layer(NeuralNetSingleOutput):
    """Holds a single layer neural network for annotating images."""

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                n_output_classes=None,
                fc1_dropout=1.0, alpha=4e-4 ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing dimensions of network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):
            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.n_output_classes = n_output_classes
            self.fc1_dropout = fc1_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_class_samples_trained = self.n_output_classes*[0]
            self.n_samples_list = []
            self.n_class_samples_list = [[] for _ in range(self.n_output_classes)]
            self.accuracy_list = [[] for _ in range(self.n_output_classes)]
            self.precision_list = [[] for _ in range(self.n_output_classes)]
            self.recall_list = [[] for _ in range(self.n_output_classes)]
            self.F1_list = [[] for _ in range(self.n_output_classes)]

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
            self.n_output_classes = net_architecture['n_output_classes']
            self.fc1_dropout = net_architecture['fc1_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_class_samples_trained = net_architecture['n_class_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.n_class_samples_list = net_architecture['n_class_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc1_dropout != fc1_dropout:
            self.fc1_dropout = fc1_dropout
            self.log("Updated dropout fraction to {}".format(self.fc1_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                    shape = [None, self.n_output_classes] )

        # Set up dropout option for inputs
        self.fc1_keep_prob = tf.placeholder(tf.float32)
        self.x_drop = tf.nn.dropout(self.x, self.fc1_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = \
            [self.y_res * self.x_res * self.n_input_channels,
                self.n_output_classes]
        self.W_fc_out = tf.Variable( tf.truncated_normal(
                                shape=self.fc_out_shape, stddev=0.1 ) )
        self.b_fc_out = tf.Variable( tf.constant(0.1,
                                shape=[self.fc_out_shape[1]] ))

        # Calculate network step
        self.fc_out_lin = tf.matmul( self.x_drop,
                                     self.W_fc_out ) + self.b_fc_out

        #########################################################
        # Define cost function and optimizer algorithm
        self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.network_prediction  = tf.cast( tf.argmax(
                                        self.fc_out_lin, 1 ), tf.float32 )
        self.is_correct_prediction = tf.equal( tf.argmax( self.fc_out_lin, 1 ),
                                               tf.argmax( self.y_trgt, 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast(
                                    self.is_correct_prediction, tf.float32 ) )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("n_output_classes: {}".format(self.n_output_classes))
        self.log("input_dropout: {}".format(self.fc1_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['n_output_classes'] = self.n_output_classes
        net_architecture['fc1_dropout'] = self.fc1_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_class_samples_trained'] = self.n_class_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['n_class_samples_list'] = self.n_class_samples_list
        net_architecture['accuracy_list'] = self.accuracy_list
        net_architecture['precision_list'] = self.precision_list
        net_architecture['recall_list'] = self.recall_list
        net_architecture['F1_list'] = self.F1_list
        np.save(os.path.join( \
            network_path,'net_architecture.npy'), net_architecture)
        self.log("Network architecture saved to file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))

    def show_filters(self):
        """Plot the input weights"""
        weight_mat = self.sess.run(self.W_fc_out)

        # Loop channels
        plt.figure(figsize=(12,5), facecolor='w', edgecolor='w')
        with sns.axes_style("white"):
            for cl in range(weight_mat.shape[1]):
                # Get filters of this output class
                w_list = ia.vec2image( lin_image=weight_mat[:,cl],
                    n_channels=self.n_input_channels,
                    image_size=(self.y_res,self.x_res) )

                # Show channels
                for ch,w in enumerate(w_list):
                    colormax = np.abs(w).max()
                    ax = plt.subplot2grid( (self.n_output_classes,
                            self.n_input_channels), (cl,ch) )
                    ax.imshow( w, interpolation='nearest',
                        cmap=plt.get_cmap('seismic'),
                        clim=(-1*colormax,colormax) )
                    ax.set_title("Class {}, Channel {}".format(cl,ch))
                    plt.axis('tight')
                    plt.axis('off')
                colormax = np.abs(w).max()

        if self.n_output_classes == 2:
            plt.figure(figsize=(12,5), facecolor='w', edgecolor='w')
            with sns.axes_style("white"):
                # Get filters of this output class
                w_list0 = ia.vec2image( lin_image=weight_mat[:,0],
                    n_channels=self.n_input_channels,
                    image_size=(self.y_res,self.x_res) )
                w_list1 = ia.vec2image( lin_image=weight_mat[:,1],
                    n_channels=self.n_input_channels,
                    image_size=(self.y_res,self.x_res) )
                for ch in range(len(w_list)):
                    w_both = w_list1[ch]-w_list0[ch]

                    colormax = np.abs(w_both).max()
                    ax = plt.subplot2grid( (1,
                            self.n_input_channels), (0,ch) )
                    ax.imshow( w_both, interpolation='nearest',
                        cmap=plt.get_cmap('seismic'),
                        clim=(-1*colormax,colormax) )
                    ax.set_title("Class {}, Channel {}".format(cl,ch))
                    plt.axis('tight')
                    plt.axis('off')
            plt.tight_layout()


########################################################################
### Two layer neural network
########################################################################

class NeuralNet2Layer(NeuralNetSingleOutput):
    """Holds a two layer neural network for annotating images."""

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                n_output_classes=None,
                fc1_n_chan=1024, fc1_dropout=0.5, alpha=4e-4 ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing dimensions of network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):
            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.n_output_classes = n_output_classes
            self.fc1_n_chan = fc1_n_chan
            self.fc1_dropout = fc1_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_class_samples_trained = self.n_output_classes*[0]
            self.n_samples_list = []
            self.n_class_samples_list = [[] for _ in range(self.n_output_classes)]
            self.accuracy_list = [[] for _ in range(self.n_output_classes)]
            self.precision_list = [[] for _ in range(self.n_output_classes)]
            self.recall_list = [[] for _ in range(self.n_output_classes)]
            self.F1_list = [[] for _ in range(self.n_output_classes)]

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
            self.n_output_classes = net_architecture['n_output_classes']
            self.fc1_n_chan = net_architecture['fc1_n_chan']
            self.fc1_dropout = net_architecture['fc1_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_class_samples_trained = net_architecture['n_class_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.n_class_samples_list = net_architecture['n_class_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc1_dropout != fc1_dropout:
            self.fc1_dropout = fc1_dropout
            self.log("Updated dropout fraction to {}".format(self.fc1_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                                    shape = [None, self.n_output_classes] )

        #########################################################
        # Densely Connected Layer
        # Weights and bias
        self.fc1_shape = \
            [self.y_res * self.x_res * self.n_input_channels,
                self.fc1_n_chan]
        self.W_fc1 = tf.Variable( tf.truncated_normal(
                               shape=self.fc1_shape, stddev=0.1 ) )
        self.b_fc1 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Calculate network step
        self.fc1_relu = tf.nn.relu( tf.matmul( self.x,
            self.W_fc1) + self.b_fc1 )

        # Set up dropout option for fc1
        self.fc1_keep_prob = tf.placeholder(tf.float32)
        self.fc1_relu_drop = tf.nn.dropout(self.fc1_relu, self.fc1_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = [self.fc1_n_chan, self.n_output_classes]
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
                    tf.nn.softmax_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.network_prediction  = tf.cast( tf.argmax(
                                        self.fc_out_lin, 1 ), tf.float32 )
        self.is_correct_prediction = tf.equal( tf.argmax( self.fc_out_lin, 1 ),
                                               tf.argmax( self.y_trgt, 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast(
                                    self.is_correct_prediction, tf.float32 ) )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("n_output_classes: {}".format(self.n_output_classes))
        self.log("fc1_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc1_dropout: {}".format(self.fc1_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['n_output_classes'] = self.n_output_classes
        net_architecture['fc1_n_chan'] = self.fc1_n_chan
        net_architecture['fc1_dropout'] = self.fc1_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_class_samples_trained'] = self.n_class_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['n_class_samples_list'] = self.n_class_samples_list
        net_architecture['accuracy_list'] = self.accuracy_list
        net_architecture['precision_list'] = self.precision_list
        net_architecture['recall_list'] = self.recall_list
        net_architecture['F1_list'] = self.F1_list
        np.save(os.path.join( \
            network_path,'net_architecture.npy'), net_architecture)
        self.log("Network architecture saved to file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))

    def show_filters(self):
        """Plot the weights of the hidden layer"""
        w_mat = np.transpose(self.sess.run(self.W_fc1))

        plt.figure(figsize=(10,10), facecolor='w', edgecolor='w')
        plot_positions = [(0,0),(0,1),(1,0),(1,1)]
        for ch in range(self.n_input_channels):
            grid,_ = ia.image_grid_RGB( w_mat,
                n_channels=self.n_input_channels,
                image_size=(self.y_res,self.x_res), n_x=6, n_y=6,
                channel_order=(ch,ch,ch), amplitude_scaling=(1,1,1),
                line_color=1, auto_scale=True, return_borders=False )
            colormax = np.abs(grid).max()
            with sns.axes_style("white"):
                ax = plt.subplot2grid( (2,2), plot_positions[ch] )
                ax.imshow( grid[:,:,0], interpolation='nearest',
                    cmap=plt.get_cmap('seismic'),
                    clim=(-1*colormax,colormax) )
                ax.set_title("Hidden units, channel {}".format(ch))
                plt.axis('tight')
                plt.axis('off')
                plt.tight_layout()


########################################################################
### Deep convolutional neural network
### 2 conv layers, one dense layer, 2 output units
########################################################################

class ConvNetCnv2Fc1(NeuralNetSingleOutput):
    """Holds a deep convolutional neural network for annotating images.
    2 convolutional layers, 1 fully connected layer, 1 output layer"""

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                n_output_classes=None,
                conv1_size=5, conv1_n_chan=32, conv1_n_pool=2,
                conv2_size=5, conv2_n_chan=64, conv2_n_pool=2,
                fc1_n_chan=1024, fc1_dropout=0.5, alpha=4e-4 ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing dimensions of network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):
            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.n_output_classes = n_output_classes
            self.conv1_size = conv1_size
            self.conv1_n_chan = conv1_n_chan
            self.conv1_n_pool = conv1_n_pool
            self.conv2_size = conv2_size
            self.conv2_n_chan = conv2_n_chan
            self.conv2_n_pool = conv2_n_pool
            self.fc1_y_size = int( np.ceil( np.ceil(
                self.y_res/self.conv1_n_pool ) / self.conv2_n_pool ) )
            self.fc1_x_size = int( np.ceil( np.ceil(
                self.x_res/self.conv1_n_pool ) / self.conv2_n_pool ) )
            self.fc1_n_chan = fc1_n_chan
            self.fc1_dropout = fc1_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_class_samples_trained = self.n_output_classes*[0]
            self.n_samples_list = []
            self.n_class_samples_list = [[] for _ in range(self.n_output_classes)]
            self.accuracy_list = [[] for _ in range(self.n_output_classes)]
            self.precision_list = [[] for _ in range(self.n_output_classes)]
            self.recall_list = [[] for _ in range(self.n_output_classes)]
            self.F1_list = [[] for _ in range(self.n_output_classes)]

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
            self.n_output_classes = net_architecture['n_output_classes']
            self.conv1_size = net_architecture['conv1_size']
            self.conv1_n_chan = net_architecture['conv1_n_chan']
            self.conv1_n_pool = net_architecture['conv1_n_pool']
            self.conv2_size = net_architecture['conv2_size']
            self.conv2_n_chan = net_architecture['conv2_n_chan']
            self.conv2_n_pool = net_architecture['conv2_n_pool']
            self.fc1_y_size = int( np.ceil( np.ceil(
                self.y_res/self.conv1_n_pool ) / self.conv2_n_pool ) )
            self.fc1_x_size = int( np.ceil( np.ceil(
                self.x_res/self.conv1_n_pool ) / self.conv2_n_pool ) )
            self.fc1_n_chan = net_architecture['fc1_n_chan']
            self.fc1_dropout = net_architecture['fc1_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_class_samples_trained = net_architecture['n_class_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.n_class_samples_list = net_architecture['n_class_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc1_dropout != fc1_dropout:
            self.fc1_dropout = fc1_dropout
            self.log("Updated dropout fraction to {}".format(self.fc1_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                                        shape = [None, self.n_output_classes] )

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
        # Convolutional layer 2
        self.conv2_shape = [self.conv2_size, self.conv2_size,
                       self.conv1_n_chan, self.conv2_n_chan]
        self.W_conv2 = tf.Variable( tf.truncated_normal(
                               shape=self.conv2_shape, stddev=0.1 ) )
        self.b_conv2 = tf.Variable( tf.constant(0.1,
                                                shape=[self.conv2_n_chan] ))

        # Convolve x_image with the weight tensor
        self.conv2_lin = tf.nn.conv2d( self.conv1_pool, self.W_conv2,
                                  strides=[1, 1, 1, 1], padding='SAME' )

        # Add bias and apply transfer function
        self.conv2_relu = tf.nn.relu( self.conv2_lin + self.b_conv2 )

        # Max pooling
        self.conv2_kernel = [1, self.conv2_n_pool, self.conv2_n_pool, 1]
        self.conv2_pool = tf.nn.max_pool( self.conv2_relu,
            ksize=self.conv2_kernel, strides=self.conv2_kernel, padding='SAME')


        #########################################################
        # Densely Connected Layer
        # Weights and bias
        self.fc1_shape = [self.fc1_y_size * self.fc1_x_size * self.conv2_n_chan,
                          self.fc1_n_chan]
        self.W_fc1 = tf.Variable( tf.truncated_normal(
                               shape=self.fc1_shape, stddev=0.1 ) )
        self.b_fc1 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Flatten output from conv2
        self.conv2_pool_flat = tf.reshape(
            self.conv2_pool, [-1, self.fc1_shape[0]] )

        # Calculate network step
        self.fc1_relu = tf.nn.relu( tf.matmul( self.conv2_pool_flat,
            self.W_fc1) + self.b_fc1 )

        # Set up dropout option for fc1
        self.fc1_keep_prob = tf.placeholder(tf.float32)
        self.fc1_relu_drop = tf.nn.dropout(self.fc1_relu, self.fc1_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = [self.fc1_n_chan, self.n_output_classes]
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
                    tf.nn.softmax_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.network_prediction  = tf.cast( tf.argmax(
                                        self.fc_out_lin, 1 ), tf.float32 )
        self.is_correct_prediction = tf.equal( tf.argmax( self.fc_out_lin, 1 ),
                                               tf.argmax( self.y_trgt, 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast(
                                    self.is_correct_prediction, tf.float32 ) )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("n_output_classes: {}".format(self.n_output_classes))
        self.log("conv1_size: {}".format(self.conv1_size))
        self.log("conv1_n_chan: {}".format(self.conv1_n_chan))
        self.log("conv1_n_pool: {}".format(self.conv1_n_pool))
        self.log("conv2_size: {}".format(self.conv2_size))
        self.log("conv2_n_chan: {}".format(self.conv2_n_chan))
        self.log("conv2_n_pool: {}".format(self.conv2_n_pool))
        self.log("fc1_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc1_dropout: {}".format(self.fc1_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['n_output_classes'] = self.n_output_classes
        net_architecture['conv1_size'] = self.conv1_size
        net_architecture['conv1_n_chan'] = self.conv1_n_chan
        net_architecture['conv1_n_pool'] = self.conv1_n_pool
        net_architecture['conv2_size'] = self.conv2_size
        net_architecture['conv2_n_chan'] = self.conv2_n_chan
        net_architecture['conv2_n_pool'] = self.conv2_n_pool
        net_architecture['fc1_n_chan'] = self.fc1_n_chan
        net_architecture['fc1_dropout'] = self.fc1_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_class_samples_trained'] = self.n_class_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['n_class_samples_list'] = self.n_class_samples_list
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

        n_iterations = 50
        return_im = np.zeros( (2,
            self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (64,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.fc1_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.conv2_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.conv1_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )

        # for filter_no in range(self.conv1_n_chan):
        # for filter_no in range(self.conv2_n_chan):
        # for filter_no in range(64):
        # for filter_no in range(self.fc1_n_chan):
        for filter_no in range(2):
            print(filter_no)

            # # Isolate activation of a single convolutional filter
            # layer_slice_begin = tf.constant( [0,0,0,filter_no], dtype=tf.int32 )
            # layer_slice_size = tf.constant( [-1,-1,-1,1], dtype=tf.int32 )
            # layer_units = tf.slice( self.conv2_relu,
            #                         layer_slice_begin, layer_slice_size )
            # # layer_units = tf.slice( self.conv1_relu,
            # #                         layer_slice_begin, layer_slice_size )

            # Isolate activation of a single fully connected unit
            layer_slice_begin = tf.constant( [0,filter_no], dtype=tf.int32 )
            layer_slice_size = tf.constant( [-1,1], dtype=tf.int32 )
            # layer_units = tf.slice( self.fc1_relu,
            #                         layer_slice_begin, layer_slice_size )
            layer_units = tf.slice( self.fc_out_lin,
                                    layer_slice_begin, layer_slice_size )

            # Define cost function for filter
            layer_activation = tf.reduce_mean( layer_units )

            # Optimize input image for maximizing filter output
            gradients = tf.gradients( ys=layer_activation, xs=[self.x] )[0]

            norm_grad = tf.nn.l2_normalize( gradients, dim=0, epsilon=1e-5)

            # Random staring point
            im = (np.random.uniform( size=(1,
                self.n_input_channels * self.y_res * self.x_res) ) * 0.1) + 0.5

            # Do a gradient ascend
            for e in range(n_iterations):
                grad = self.sess.run( [norm_grad], feed_dict={
                                self.x: im, self.fc1_keep_prob: 1.0 } )[0]
                # Gradient ASCENT
                im += 0.5*grad

            im -= im.mean()
            im /= (im.std() + 1e-5)
            im *= 0.2

            # clip to [0, 1]
            im += 0.5
            im = np.clip(im, 0, 1)
            return_im[filter_no,:] = im

        plt.figure(figsize=(16.5,9), facecolor='w', edgecolor='w')
        for ch in range(self.n_input_channels):

            grid_im,_,brdr = ia.image_grid_RGB( return_im,
                n_channels=self.n_input_channels,
                image_size=(self.y_res,self.x_res), n_x=16, n_y=4,
                channel_order=(ch,ch,ch), amplitude_scaling=(1,1,1),
                line_color=1, auto_scale=True, return_borders=True )

            ax = plt.subplot2grid( (self.n_input_channels,1), (ch,0) )
            with sns.axes_style("white"):
                ax.imshow( grid_im,
                    interpolation='nearest', vmax=grid_im.max() )
                ax.set_title("Filterss, ch {}".format(ch))
                plt.axis('tight')
                plt.axis('off')

        if self.n_input_channels == 3:
            channel_selector = (0,1,2)
            chan_ampl = (1,1,1)
        elif self.n_input_channels == 2:
            channel_selector = (0,1,1)
            chan_ampl = (1,1,0)
        elif self.n_input_channels == 1:
            channel_selector = (0,0,0)
            chan_ampl = (1,1,1)

        grid_im,_,brdr = ia.image_grid_RGB( return_im,
            n_channels=self.n_input_channels,
            image_size=(self.y_res,self.x_res), n_x=8, n_y=8,
            channel_order=channel_selector, amplitude_scaling=chan_ampl,
            line_color=1, auto_scale=True, return_borders=True )
        fig, ax = plt.subplots(figsize=(9,9), facecolor='w', edgecolor='w')
        with sns.axes_style("white"):
            ax.imshow( grid_im,
                interpolation='nearest', vmax=grid_im.max() )
            ax.set_title("Filters RGB")
            plt.axis('tight')
            plt.axis('off')

        plt.tight_layout()


########################################################################
### Deep convolutional neural network
### N conv layers, N_ dense layers, 1 output layer
########################################################################

class ConvNetCnvNFc1(NeuralNetSingleOutput):
    """Holds a deep convolutional neural network for annotating images.
    N convolutional layers, 1 fully connected layer, 1 output layer"""

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                n_output_classes=None,
                conv_n_layers=3, conv_size=5, conv_n_chan=32, conv_n_pool=2,
                fc1_n_chan=1024, fc1_dropout=0.5, alpha=4e-4,
                reduced_output=False ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing dimensions of network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):
            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.n_output_classes = n_output_classes
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
            self.fc1_dropout = fc1_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_class_samples_trained = self.n_output_classes*[0]
            self.n_samples_list = []
            self.n_class_samples_list = [[] for _ in range(self.n_output_classes)]
            self.accuracy_list = [[] for _ in range(self.n_output_classes)]
            self.precision_list = [[] for _ in range(self.n_output_classes)]
            self.recall_list = [[] for _ in range(self.n_output_classes)]
            self.F1_list = [[] for _ in range(self.n_output_classes)]

            # Save network architecture
            self.save_network_architecture( network_path=self.network_path )

        else:
            now = datetime.datetime.now()
            if reduced_output:
                self.log( "Initializing {} @ {}".format( \
                    self.network_path, now.strftime( "%Y-%m-%d %H:%M" ) ) )
            else:
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
            self.n_output_classes = net_architecture['n_output_classes']
            self.conv_n_layers = net_architecture['conv_n_layers']
            self.conv_size = net_architecture['conv_size']
            self.conv_n_chan = net_architecture['conv_n_chan']
            self.conv_n_pool = net_architecture['conv_n_pool']

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
            self.fc1_dropout = net_architecture['fc1_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_class_samples_trained = net_architecture['n_class_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.n_class_samples_list = net_architecture['n_class_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            if reduced_output is not True:
                self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc1_dropout != fc1_dropout:
            self.fc1_dropout = fc1_dropout
            if reduced_output is not True:
                self.log("Updated dropout fraction to {}".format(self.fc1_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                                        shape = [None, self.n_output_classes] )

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
        self.fc_out_shape = [self.fc1_n_chan, self.n_output_classes]
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
                    tf.nn.softmax_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.network_prediction  = tf.cast( tf.argmax(
                                        self.fc_out_lin, 1 ), tf.float32 )
        self.is_correct_prediction = tf.equal( tf.argmax( self.fc_out_lin, 1 ),
                                               tf.argmax( self.y_trgt, 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast(
                                    self.is_correct_prediction, tf.float32 ) )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("n_output_classes: {}".format(self.n_output_classes))
        self.log("conv_n_layers: {}".format(self.conv_n_layers))
        for L in range(self.conv_n_layers):
            self.log("conv{}_size: {}".format(L+1,self.conv_size))
            self.log("conv{}_n_input_chan: {}".format(L+1,self.conv_n_chan_L[L]))
            self.log("conv{}_n_output_chan: {}".format(L+1,self.conv_n_chan_L[L+1]))
            self.log("conv{}_n_pool: {}".format(L+1,self.conv_n_pool))
        self.log("fc1_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc1_dropout: {}".format(self.fc1_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['n_output_classes'] = self.n_output_classes
        net_architecture['conv_n_layers'] = self.conv_n_layers
        net_architecture['conv_size'] = self.conv_size
        net_architecture['conv_n_chan'] = self.conv_n_chan
        net_architecture['conv_n_pool'] = self.conv_n_pool
        net_architecture['fc1_n_chan'] = self.fc1_n_chan
        net_architecture['fc1_dropout'] = self.fc1_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_class_samples_trained'] = self.n_class_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['n_class_samples_list'] = self.n_class_samples_list
        net_architecture['accuracy_list'] = self.accuracy_list
        net_architecture['precision_list'] = self.precision_list
        net_architecture['recall_list'] = self.recall_list
        net_architecture['F1_list'] = self.F1_list
        np.save(os.path.join( \
            network_path,'net_architecture.npy'), net_architecture)
        self.log("Network architecture saved to file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))

    def show_conv_filters(self):
        """Plot the convolutional filters"""

        # Define which filters to optimize
        filter_layer = self.conv_n_layers-1
        n_filters = int(np.min((self.conv_n_chan_L[filter_layer+1],96)))
        n_iterations = 50
        filter_ims = np.zeros( (n_filters,
            self.n_input_channels * self.y_res * self.x_res) )

        if self.n_input_channels == 3:
            channel_selector = (0,1,2)
            chan_ampl = (1,1,1)
        elif self.n_input_channels == 2:
            channel_selector = (0,1,1)
            chan_ampl = (1,1,0)
        elif self.n_input_channels == 1:
            channel_selector = (0,0,0)
            chan_ampl = (1,1,1)

        # Loop all filters
        print("Maximizing output of {} filters: {:3d}".format(n_filters,0),
                end="", flush=True)
        for filter_no in range(n_filters):
            print((3*'\b')+'{:3d}'.format(filter_no), end='', flush=True)

            # Isolate activation of a single convolutional filter
            layer_slice_begin = tf.constant( [0,0,0,filter_no], dtype=tf.int32 )
            layer_slice_size = tf.constant( [-1,-1,-1,1], dtype=tf.int32 )
            layer_units = tf.slice( self.conv_relu[filter_layer],
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
            for e in range(n_iterations-1):
                result = self.sess.run( [norm_grad], feed_dict={ self.x: im } )
                im += result[0]
            filter_ims[filter_no,:] = im

        print((3*'\b')+'{:3d}'.format(filter_no))

        # Create grid with filters
        grid_im,_,brdr = ia.image_grid_RGB( filter_ims,
            n_channels=self.n_input_channels,
            image_size=(self.y_res,self.x_res), n_x=12, n_y=8,
            channel_order=channel_selector, amplitude_scaling=chan_ampl,
            line_color=1, auto_scale=True, return_borders=True )

        # Plot
        with sns.axes_style("white"):
            fig,ax = plt.subplots(figsize=(12,8), facecolor='w', edgecolor='w')
            ax.imshow( grid_im,
                interpolation='nearest', vmax=grid_im.max() )
            ax.set_title("Filters, conv layer {}".format(filter_layer+1))
            plt.axis('tight')
            plt.axis('off')
        plt.tight_layout()

    def show_filters(self):
        """Plot the convolutional filters"""

        n_iterations = 50
        return_im = np.zeros( (2,
            self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (64,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.fc1_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.conv2_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.conv1_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        if self.n_input_channels == 3:
            channel_selector = (0,1,2)
            chan_ampl = (1,1,1)
        elif self.n_input_channels == 2:
            channel_selector = (0,1,1)
            chan_ampl = (1,1,0)
        elif self.n_input_channels == 1:
            channel_selector = (0,0,0)
            chan_ampl = (1,1,1)

        # for filter_no in range(self.conv1_n_chan):
        # for filter_no in range(self.conv2_n_chan):
        # for filter_no in range(64):
        # for filter_no in range(self.fc1_n_chan):
        for filter_no in range(2):
            print(filter_no)

            # # Isolate activation of a single convolutional filter
            # layer_slice_begin = tf.constant( [0,0,0,filter_no], dtype=tf.int32 )
            # layer_slice_size = tf.constant( [-1,-1,-1,1], dtype=tf.int32 )
            # layer_units = tf.slice( self.conv2_relu,
            #                         layer_slice_begin, layer_slice_size )
            # # layer_units = tf.slice( self.conv1_relu,
            # #                         layer_slice_begin, layer_slice_size )

            # Isolate activation of a single fully connected unit
            layer_slice_begin = tf.constant( [0,filter_no], dtype=tf.int32 )
            layer_slice_size = tf.constant( [-1,1], dtype=tf.int32 )
            # layer_units = tf.slice( self.fc1_relu,
            #                         layer_slice_begin, layer_slice_size )
            layer_units = tf.slice( self.fc_out_lin,
                                    layer_slice_begin, layer_slice_size )

            # Define cost function for filter
            layer_activation = tf.reduce_mean( layer_units )

            # Optimize input image for maximizing filter output
            gradients = tf.gradients( ys=layer_activation, xs=[self.x] )[0]

            norm_grad = tf.nn.l2_normalize( gradients, dim=0, epsilon=1e-5)

            # Random staring point
            im = (np.random.uniform( size=(1,
                self.n_input_channels * self.y_res * self.x_res) ) * 0.1) + 0.5

            # Do a gradient ascend
            for e in range(n_iterations):
                grad = self.sess.run( [norm_grad], feed_dict={
                                self.x: im, self.fc1_keep_prob: 1.0 } )[0]
                # Gradient ASCENT
                im += 0.5*grad

            im -= im.mean()
            im /= (im.std() + 1e-5)
            im *= 0.2

            # clip to [0, 1]
            im += 0.5
            im = np.clip(im, 0, 1)
            return_im[filter_no,:] = im

        plt.figure(figsize=(16.5,9), facecolor='w', edgecolor='w')
        for ch in range(self.n_input_channels):

            grid_im,_,brdr = ia.image_grid_RGB( return_im,
                n_channels=self.n_input_channels,
                image_size=(self.y_res,self.x_res), n_x=16, n_y=4,
                channel_order=(ch,ch,ch), amplitude_scaling=(1,1,1),
                line_color=1, auto_scale=True, return_borders=True )

            ax = plt.subplot2grid( (self.n_input_channels,1), (ch,0) )
            with sns.axes_style("white"):
                ax.imshow( grid_im,
                    interpolation='nearest', vmax=grid_im.max() )
                ax.set_title("Filterss, ch {}".format(ch))
                plt.axis('tight')
                plt.axis('off')

        grid_im,_,brdr = ia.image_grid_RGB( return_im,
            n_channels=self.n_input_channels,
            image_size=(self.y_res,self.x_res), n_x=8, n_y=8,
            channel_order=channel_selector, amplitude_scaling=chan_ampl,
            line_color=1, auto_scale=True, return_borders=True )
        fig, ax = plt.subplots(figsize=(9,9), facecolor='w', edgecolor='w')
        with sns.axes_style("white"):
            ax.imshow( grid_im,
                interpolation='nearest', vmax=grid_im.max() )
            ax.set_title("Filters RGB")
            plt.axis('tight')
            plt.axis('off')

        plt.tight_layout()


########################################################################
### Deep convolutional neural network
### N conv layers, N_ dense layers, 1 output layer
########################################################################

class ConvNetCnvNFc2(NeuralNetSingleOutput):
    """Holds a deep convolutional neural network for annotating images. N convolutional layers, 2 identical fully connected layers, 1 output layer """

    def __init__(self, network_path='.', logging=True,
                input_image_size=None, n_input_channels=None,
                n_output_classes=None,
                conv_n_layers=3, conv_size=5, conv_n_chan=32, conv_n_pool=2,
                fc1_n_chan=1024, fc1_dropout=0.5, alpha=4e-4,
                reduced_output=False ):
        """Initializes all variables and sets up the network. If network
        already exists, load the variables from there.
        network_path:      Directory where to store network and architecture
        input_image_size:  Tuple containing (y,x) size of input image
        output_image_size: Tuple containing dimensions of network output"""
        self.logging = logging

        # If network path does not yet exists
        self.network_path = network_path
        if not os.path.isdir(self.network_path):
            # Make network directory
            os.mkdir(self.network_path)
            now = datetime.datetime.now()
            self.log("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log(    "Creation of new network: ")
            self.log(    "  {}".format(self.network_path) )
            self.log(    "  @ {}".format(now.strftime("%Y-%m-%d %H:%M")) )
            self.log(    "++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("\nNetwork did not exist ... ")
            self.log("Created new network with supplied (or default) architecture")

            # Set up new network
            self.y_res = input_image_size[0]
            self.x_res = input_image_size[1]
            self.n_input_channels = n_input_channels
            self.n_output_classes = n_output_classes
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
            self.fc1_dropout = fc1_dropout
            self.alpha = alpha
            self.n_samples_trained = 0
            self.n_class_samples_trained = self.n_output_classes*[0]
            self.n_samples_list = []
            self.n_class_samples_list = [[] for _ in range(self.n_output_classes)]
            self.accuracy_list = [[] for _ in range(self.n_output_classes)]
            self.precision_list = [[] for _ in range(self.n_output_classes)]
            self.recall_list = [[] for _ in range(self.n_output_classes)]
            self.F1_list = [[] for _ in range(self.n_output_classes)]

            # Save network architecture
            self.save_network_architecture( network_path=self.network_path )

        else:
            now = datetime.datetime.now()
            if reduced_output:
                self.log( "Initializing {} @ {}".format( \
                    self.network_path, now.strftime( "%Y-%m-%d %H:%M" ) ) )
            else:
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
            self.n_output_classes = net_architecture['n_output_classes']
            self.conv_n_layers = net_architecture['conv_n_layers']
            self.conv_size = net_architecture['conv_size']
            self.conv_n_chan = net_architecture['conv_n_chan']
            self.conv_n_pool = net_architecture['conv_n_pool']

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
            self.fc1_dropout = net_architecture['fc1_dropout']
            self.alpha = net_architecture['alpha']
            self.n_samples_trained = net_architecture['n_samples_trained']
            self.n_class_samples_trained = net_architecture['n_class_samples_trained']
            self.n_samples_list = net_architecture['n_samples_list']
            self.n_class_samples_list = net_architecture['n_class_samples_list']
            self.accuracy_list = net_architecture['accuracy_list']
            self.precision_list = net_architecture['precision_list']
            self.recall_list = net_architecture['recall_list']
            self.F1_list = net_architecture['F1_list']

        # Update values of alpha and dropout if supplied
        if self.alpha != alpha:
            self.alpha = alpha
            if reduced_output is not True:
                self.log("Updated learning rate 'alpha' to {}".format(self.alpha))
        if self.fc1_dropout != fc1_dropout:
            self.fc1_dropout = fc1_dropout
            if reduced_output is not True:
                self.log("Updated dropout fraction to {}".format(self.fc1_dropout))

        # Clear previous graphs
        tf.reset_default_graph()

        #########################################################
        # Input and target variable placeholders
        # x = [ m_samples x [channel_1_data, channel_2_data, etc.] ]
        self.x = tf.placeholder( tf.float32, shape = [None,
            self.n_input_channels * self.y_res * self.x_res] )
        self.y_trgt = tf.placeholder( tf.float32, \
                                        shape = [None, self.n_output_classes] )

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
        # Densely Connected Layer 1
        # Weights and bias
        self.fc1_shape = [self.fc1_y_size * self.fc1_x_size * self.conv_n_chan_L[-1], self.fc1_n_chan]
        self.W_fc1 = tf.Variable( tf.truncated_normal(
                               shape=self.fc1_shape, stddev=0.1 ) )
        self.b_fc1 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Flatten output from convN
        self.conv_last_pool_flat = tf.reshape(
            self.conv_pool[-1], [-1, self.fc1_shape[0]] )

        # Calculate network step
        self.fc1_relu = tf.nn.relu( tf.matmul( self.conv_last_pool_flat,
            self.W_fc1) + self.b_fc1 )

        #########################################################
        # Densely Connected Layer 2
        # Weights and bias
        self.fc2_shape = [self.fc1_n_chan, self.fc1_n_chan]

        self.W_fc2 = tf.Variable( tf.truncated_normal(
                               shape=self.fc2_shape, stddev=0.1 ) )
        self.b_fc2 = tf.Variable( tf.constant(0.1, shape=[self.fc1_n_chan] ))

        # Calculate network step
        self.fc2_relu = tf.nn.relu( tf.matmul( self.fc1_relu,
            self.W_fc2) + self.b_fc2 )

        # Set up dropout option for fc1
        self.fc1_keep_prob = tf.placeholder(tf.float32)
        self.fc2_relu_drop = tf.nn.dropout(self.fc2_relu, self.fc1_keep_prob)

        #########################################################
        # Readout layer
        # Weights and bias
        self.fc_out_shape = [self.fc1_n_chan, self.n_output_classes]
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
                    tf.nn.softmax_cross_entropy_with_logits(
                                logits=self.fc_out_lin, labels=self.y_trgt ) )
        self.train_step = tf.train.AdamOptimizer(self.alpha).minimize(
                                                        self.cross_entropy )

        #########################################################
        # Define how to test trained model
        self.network_prediction  = tf.cast( tf.argmax(
                                        self.fc_out_lin, 1 ), tf.float32 )
        self.is_correct_prediction = tf.equal( tf.argmax( self.fc_out_lin, 1 ),
                                               tf.argmax( self.y_trgt, 1 ) )
        self.accuracy = tf.reduce_mean( tf.cast(
                                    self.is_correct_prediction, tf.float32 ) )

        #########################################################
        # Create save operation
        self.saver = tf.train.Saver()

    def display_network_architecture(self):
        """Displays the network architecture"""
        self.log("\n-------- Network architecture --------")
        self.log("y_res: {}".format(self.y_res))
        self.log("x_res: {}".format(self.x_res))
        self.log("n_input_channels: {}".format(self.n_input_channels))
        self.log("n_output_classes: {}".format(self.n_output_classes))
        self.log("conv_n_layers: {}".format(self.conv_n_layers))
        for L in range(self.conv_n_layers):
            self.log("conv{}_size: {}".format(L+1,self.conv_size))
            self.log("conv{}_n_input_chan: {}".format(L+1,self.conv_n_chan_L[L]))
            self.log("conv{}_n_output_chan: {}".format(L+1,self.conv_n_chan_L[L+1]))
            self.log("conv{}_n_pool: {}".format(L+1,self.conv_n_pool))
        self.log("fc1_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc2_n_chan: {}".format(self.fc1_n_chan))
        self.log("fc2_dropout: {}".format(self.fc1_dropout))
        self.log("alpha: {}".format(self.alpha))
        self.log("n_samples_trained: {}".format(self.n_samples_trained))
        for c in range(self.n_output_classes):
            self.log( " * Class {}, m = {}".format( \
                c, self.n_class_samples_trained[c] ) )

    def save_network_architecture(self,network_path):
        """Saves the network architecture into the network path"""
        net_architecture = {}
        net_architecture['y_res'] = self.y_res
        net_architecture['x_res'] = self.x_res
        net_architecture['n_input_channels'] = self.n_input_channels
        net_architecture['n_output_classes'] = self.n_output_classes
        net_architecture['conv_n_layers'] = self.conv_n_layers
        net_architecture['conv_size'] = self.conv_size
        net_architecture['conv_n_chan'] = self.conv_n_chan
        net_architecture['conv_n_pool'] = self.conv_n_pool
        net_architecture['fc1_n_chan'] = self.fc1_n_chan
        net_architecture['fc1_dropout'] = self.fc1_dropout
        net_architecture['alpha'] = self.alpha
        net_architecture['n_samples_trained'] = self.n_samples_trained
        net_architecture['n_class_samples_trained'] = self.n_class_samples_trained
        net_architecture['n_samples_list'] = self.n_samples_list
        net_architecture['n_class_samples_list'] = self.n_class_samples_list
        net_architecture['accuracy_list'] = self.accuracy_list
        net_architecture['precision_list'] = self.precision_list
        net_architecture['recall_list'] = self.recall_list
        net_architecture['F1_list'] = self.F1_list
        np.save(os.path.join( \
            network_path,'net_architecture.npy'), net_architecture)
        self.log("Network architecture saved to file:\n{}".format(
                            os.path.join(network_path,'net_architecture.npy')))

    def show_conv_filters(self):
        """Plot the convolutional filters"""

        # Define which filters to optimize
        filter_layer = self.conv_n_layers-1
        n_filters = int(np.min((self.conv_n_chan_L[filter_layer+1],96)))
        n_iterations = 50
        filter_ims = np.zeros( (n_filters,
            self.n_input_channels * self.y_res * self.x_res) )

        if self.n_input_channels == 3:
            channel_selector = (0,1,2)
            chan_ampl = (1,1,1)
        elif self.n_input_channels == 2:
            channel_selector = (0,1,1)
            chan_ampl = (1,1,0)
        elif self.n_input_channels == 1:
            channel_selector = (0,0,0)
            chan_ampl = (1,1,1)

        # Loop all filters
        print("Maximizing output of {} filters: {:3d}".format(n_filters,0),
                end="", flush=True)
        for filter_no in range(n_filters):
            print((3*'\b')+'{:3d}'.format(filter_no), end='', flush=True)

            # Isolate activation of a single convolutional filter
            layer_slice_begin = tf.constant( [0,0,0,filter_no], dtype=tf.int32 )
            layer_slice_size = tf.constant( [-1,-1,-1,1], dtype=tf.int32 )
            layer_units = tf.slice( self.conv_relu[filter_layer],
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
            for e in range(n_iterations-1):
                result = self.sess.run( [norm_grad], feed_dict={ self.x: im } )
                im += result[0]
            filter_ims[filter_no,:] = im

        print((3*'\b')+'{:3d}'.format(filter_no))

        # Create grid with filters
        grid_im,_,brdr = ia.image_grid_RGB( filter_ims,
            n_channels=self.n_input_channels,
            image_size=(self.y_res,self.x_res), n_x=12, n_y=8,
            channel_order=channel_selector, amplitude_scaling=chan_ampl,
            line_color=1, auto_scale=True, return_borders=True )

        # Plot
        with sns.axes_style("white"):
            fig,ax = plt.subplots(figsize=(12,8), facecolor='w', edgecolor='w')
            ax.imshow( grid_im,
                interpolation='nearest', vmax=grid_im.max() )
            ax.set_title("Filters, conv layer {}".format(filter_layer+1))
            plt.axis('tight')
            plt.axis('off')
        plt.tight_layout()

    def show_filters(self):
        """Plot the convolutional filters"""

        n_iterations = 50
        return_im = np.zeros( (2,
            self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (64,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.fc1_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.conv2_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        # return_im = np.zeros( (self.conv1_n_chan,
        #     self.n_input_channels * self.y_res * self.x_res) )
        if self.n_input_channels == 3:
            channel_selector = (0,1,2)
            chan_ampl = (1,1,1)
        elif self.n_input_channels == 2:
            channel_selector = (0,1,1)
            chan_ampl = (1,1,0)
        elif self.n_input_channels == 1:
            channel_selector = (0,0,0)
            chan_ampl = (1,1,1)

        # for filter_no in range(self.conv1_n_chan):
        # for filter_no in range(self.conv2_n_chan):
        # for filter_no in range(64):
        # for filter_no in range(self.fc1_n_chan):
        for filter_no in range(2):
            print(filter_no)

            # # Isolate activation of a single convolutional filter
            # layer_slice_begin = tf.constant( [0,0,0,filter_no], dtype=tf.int32 )
            # layer_slice_size = tf.constant( [-1,-1,-1,1], dtype=tf.int32 )
            # layer_units = tf.slice( self.conv2_relu,
            #                         layer_slice_begin, layer_slice_size )
            # # layer_units = tf.slice( self.conv1_relu,
            # #                         layer_slice_begin, layer_slice_size )

            # Isolate activation of a single fully connected unit
            layer_slice_begin = tf.constant( [0,filter_no], dtype=tf.int32 )
            layer_slice_size = tf.constant( [-1,1], dtype=tf.int32 )
            # layer_units = tf.slice( self.fc1_relu,
            #                         layer_slice_begin, layer_slice_size )
            layer_units = tf.slice( self.fc_out_lin,
                                    layer_slice_begin, layer_slice_size )

            # Define cost function for filter
            layer_activation = tf.reduce_mean( layer_units )

            # Optimize input image for maximizing filter output
            gradients = tf.gradients( ys=layer_activation, xs=[self.x] )[0]

            norm_grad = tf.nn.l2_normalize( gradients, dim=0, epsilon=1e-5)

            # Random staring point
            im = (np.random.uniform( size=(1,
                self.n_input_channels * self.y_res * self.x_res) ) * 0.1) + 0.5

            # Do a gradient ascend
            for e in range(n_iterations):
                grad = self.sess.run( [norm_grad], feed_dict={
                                self.x: im, self.fc1_keep_prob: 1.0 } )[0]
                # Gradient ASCENT
                im += 0.5*grad

            im -= im.mean()
            im /= (im.std() + 1e-5)
            im *= 0.2

            # clip to [0, 1]
            im += 0.5
            im = np.clip(im, 0, 1)
            return_im[filter_no,:] = im

        plt.figure(figsize=(16.5,9), facecolor='w', edgecolor='w')
        for ch in range(self.n_input_channels):

            grid_im,_,brdr = ia.image_grid_RGB( return_im,
                n_channels=self.n_input_channels,
                image_size=(self.y_res,self.x_res), n_x=16, n_y=4,
                channel_order=(ch,ch,ch), amplitude_scaling=(1,1,1),
                line_color=1, auto_scale=True, return_borders=True )

            ax = plt.subplot2grid( (self.n_input_channels,1), (ch,0) )
            with sns.axes_style("white"):
                ax.imshow( grid_im,
                    interpolation='nearest', vmax=grid_im.max() )
                ax.set_title("Filterss, ch {}".format(ch))
                plt.axis('tight')
                plt.axis('off')

        grid_im,_,brdr = ia.image_grid_RGB( return_im,
            n_channels=self.n_input_channels,
            image_size=(self.y_res,self.x_res), n_x=8, n_y=8,
            channel_order=channel_selector, amplitude_scaling=chan_ampl,
            line_color=1, auto_scale=True, return_borders=True )
        fig, ax = plt.subplots(figsize=(9,9), facecolor='w', edgecolor='w')
        with sns.axes_style("white"):
            ax.imshow( grid_im,
                interpolation='nearest', vmax=grid_im.max() )
            ax.set_title("Filters RGB")
            plt.axis('tight')
            plt.axis('off')

        plt.tight_layout()
