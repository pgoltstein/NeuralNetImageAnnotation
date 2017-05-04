# Default Settings
network_path = '/nn'
training_data_path = '/data/EGR/Dataset1'
# network_path = 'E:/data/NeuralNets'
# training_data_path = 'E:/data/Dataset1'

# Annotation arguments
annotation_size = 43 # Size of the images around the annotations
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
n_epochs = 40
m_samples = 1000
number_of_batches = 10
batch_size = 1000
report_every = 10
fc1_dropout = 0.5 # Keep-fraction in last fully connectd layer during training
alpha = 0.0001 # Learning rate (typically smaller than 0.001)

# Network arguments
network_type = "c2fc1" # Neural net type "1layer", "2layer", "c2fc1"
conv_size = 9 # Size of convolutional filters (if applicable)
conv_chan = 16 # Number of convolution channels (if applicable)
conv_pool = 2 # Number of channels to pool after conv-step  (if applicable)
fc_size = 128 # Number of units in first fully connected layer (if applicable)

# Other variables
exclude_border = (40,40,40,40) # (Left, Right, Top, Border) margin, or "Load" from file
normalize_images = True # True/False whether to normalize entire images to max
rotation_list = (0,360,1) # Angle (min,max,interval)
scale_list_x = (0.9,1.1,0.01) # Scale factor (min,max,interval)
scale_list_y = (0.9,1.1,0.01) # Scale factor (min,max,interval)
noise_level_list = (0.0,0.1,0.01) # Noise stdev (min,max,interval)
