# Default Settings
network_path = '/nn'
training_data_path = '/data/Dataset_small'
# network_path = 'E:/data/NeuralNets'
# training_data_path = 'E:/data/Dataset1'

# Annotation arguments
selection_type = "Bodies"
segment_all = True
image_size = 31 # Size of the images around the annotations
annotation_size = 21 # Size of the annotations to be made
morph_annotations = False # Random morphing of annotations
include_annotation_typenrs = 1 # Include only annotations of certain type_nr
centroid_dilation_factor = 2 # Dilation/erosion (negative) of annotation centroid
body_dilation_factor = 0 # Dilation/erosion (negative) of annotation body
sample_ratio = None # Fraction samples per class (None=all equal).
annotation_border_ratio = None # Fraction from border between pos and neg samples
use_channels = [1,2,4] # List of channels that are to be used, 1 based e.g. [1,2]
normalize_samples = False # True/False
downsample_image = None # downsampling of image by factor of #

# Training arguments
training_procedure = "batch" # "batch" or "epochs"
n_epochs = 10
m_samples = 1000
number_of_batches = 10
batch_size = 1500
report_every = 5
fc_dropout = 0.5 # Keep-fraction in last fully connectd layer during training
alpha = 0.0005 # Learning rate (typically smaller than 0.001)

# Network arguments
network_type = "cNfcN" # Neural net type "cNfcN" is the only option for now
conv_n_layers = 4 # Number of convolutional layers
conv_size = 3 # Size of convolutional filters
conv_chan = 32 # Number of convolution channels
conv_pool = 2 # Number of channels to pool after conv-step
fc_n_layers = 1 # Number of fully connected layers
fc_size = 2048 # Number of units in each fully connected layer

# Other variables
exclude_border = (40,40,40,40) # (Left, Right, Top, Border) margin, or "Load" from file
normalize_images = True # True/False whether to normalize entire images to max
rotation_list = (0,360,1) # Angle (min,max,interval)
scale_list_x = (0.8,1.2,0.01) # Scale factor (min,max,interval)
scale_list_y = (0.8,1.2,0.01) # Scale factor (min,max,interval)
noise_level_list = (0.1,0.3,0.01) # Noise stdev (min,max,interval)
