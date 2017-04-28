# Default Settings
network_path = '/nn/'
training_data_path = '/data/Dataset_small'
# network_path = 'E:/data/NeuralNets'
# training_data_path = 'E:/data/Dataset1'

# Annotation arguments
selection_type = "Bodies"
segment_all = True
annotation_size = 36 # Size of the images around the annotations
morph_annotations = False # Random morphing of annotations
include_annotation_typenrs = 1 # Include only annotations of certain type_nr
centroid_dilation_factor = 3 # Dilation/erosion (negative) of annotation centroid
body_dilation_factor = 0 # Dilation/erosion (negative) of annotation body
sample_ratio = None # Fraction samples per class (None=all equal).
annotation_border_ratio = 0.2 # Fraction from border between pos and neg samples
use_channels = [1,2,3,4] # List of channels that are to be used, 1 based e.g. [1,2]
normalize_samples = False # True/False
downsample_image = None # downsampling of image by factor of #

# Training arguments
training_procedure = "batch" # "batch" or "epochs"
n_epochs = 10
m_samples = 1000
number_of_batches = 100
batch_size = 1000
report_every = 5
fc1_dropout = 1.0 # Keep-fraction in last fully connectd layer during training
alpha = 0.0002 # Learning rate (typically smaller than 0.001)

# Network arguments
network_type = "c2fc1" # Neural net type "1layer", "2layer", "c2fc1"
conv_size = 3 # Size of convolutional filters (if applicable)
conv_chan = 16 # Number of convolution channels (if applicable)
conv_pool = 1 # Number of channels to pool after conv-step  (if applicable)
fc_size = 2048 # Number of units in first fully connected layer (if applicable)

# Other variables
exclude_border = (40,40,40,40) # (Left, Right, Top, Border) margin, or "Load" from file
normalize_images = True # True/False whether to normalize entire images to max
rotation_list = (0,360,1) # Angle (min,max,interval)
scale_list_x = (0.8,1.2,0.01) # Scale factor (min,max,interval)
scale_list_y = (0.8,1.2,0.01) # Scale factor (min,max,interval)
noise_level_list = (0.1,0.3,0.01) # Noise stdev (min,max,interval)