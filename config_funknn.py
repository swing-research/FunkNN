epochs_funknn = 200 # number of epochs to train funknn network
batch_size = 64
dataset = 'celeb-hq'
gpu_num = 0 # GPU number
exp_desc = 'default' # Add a small descriptor to the experiment
image_size = 128 # Maximum resolution of the training dataset
c = 3 # Number of channels of the dataset
train_funknn = True # Train or just reload to test
restore_funknn = True
training_mode = 'factor' # Training modes for funknn: {conitinuous, factor, single}
ood_analysis = True # Evaluating the performance of model over out of distribution data (Lsun-bedroom)
interpolation_kernel = 'bicubic' # interpolation kernels : {'cubic_conv', 'bilinear', 'bicubic'}
# interpolation_kernel = 'bicubic' cannot be used for solving PDEs as its derivatives are not computed
# but can be safely used for super-resolution task.
# interpolation_kernel = 'bilinear' is fast but can only be used for PDEs with the fiesr-order derivatives.
# interpolation_kernel = 'cubic_conv' is slow but can be used for solving PDEs with first- and second-order derivatives 
network = 'MLP' # The network can be a 'CNN' or 'MLP' ('MLP' is supposed to be significantly faster)
activation = 'relu' # Activation function 'relu' or 'sin' ('sin' for more accurate spatial derivatives)

# Evaluation arguements
max_scale = 2 # Maximum scale to generate in test time (2 or 4 or 8) (=<8 for celeba-hq and 2 for other datasets)
recursive = True # Recursive image reconstructions (Use just for factor training mode)
sample_number = 25 # Number of samples in evaluation
derivatives_evaluation = False # To evaluate the performance of the model for computing the derivatives
# (Keep it False for 'bicubic' kernel)

# Datasets paths:
ood_path = 'datasets/lsun_bedroom_val'
train_path = 'datasets/celeba_hq/celeba_hq_1024_train/'
test_path = 'datasets/celeba_hq/celeba_hq_1024_test/'
