# Default Settings
network_path = '/Users/pgoltstein/Dropbox/NeuralNets'
training_data_path = '/Users/pgoltstein/Dropbox/TEMP/DataSet_small'

# Annotation arguments
annotation_size = 27 # Size of the images around the annotations
morph_annotations = False # Random morphing of annotations
include_annotation_typenrs = [1,4] # Include only annotations of certain type_nr
dilation_factor = None # Dilation/erosion (negative) of annotation body/centroid
sample_ratio = [0.45,0.45,0.1] # Fraction positive samples (of all samples)
annotation_border_ratio = None # Fraction from border between pos and neg samples
use_channels = None # List of channels that are to be used, 1 based e.g. [1,2]
normalize_samples = False # True/False

# Training arguments
training_procedure = "batch" # "batch" or "epochs"
n_epochs = 100
m_samples = 1000
number_of_batches = 10
batch_size = 1000
report_every = 10
fc1_dropout = 0.5 # Keep-fraction in last fully connectd layer during training
alpha = 0.0002 # Learning rate (typically smaller than 0.001)

# Network arguments
network_type = "c2fc1" # Neural net type "1layer", "2layer", "c2fc1"
conv_size = 5 # Size of convolutional filters (if applicable)
conv_chan = 16 # Number of convolution channels (if applicable)
conv_pool = 2 # Number of channels to pool after conv-step  (if applicable)
fc_size = 64 # Number of units in first fully connected layer (if applicable)

# Other variables
# exclude_border = (0,0,0,0) # (Left, Right, Top, Border) margin
exclude_border = (40,40,40,40) # (Left, Right, Top, Border) margin
normalize_images = True # True/False whether to normalize entire images to max
rotation_list = (0,360,1) # Angle (min,max,interval)
scale_list_x = (0.9,1.1,0.01) # Scale factor (min,max,interval)
scale_list_y = (0.9,1.1,0.01) # Scale factor (min,max,interval)
noise_level_list = (0,0.05,0.001) # Noise stdev (min,max,interval)
