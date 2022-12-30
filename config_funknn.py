epochs_funknn = 200 # number of epochs to train funknn network
batch_size = 64
dataset = 'celeb-hq'
gpu_num = 0 # GPU number
exp_desc = 'default' # Add a small descriptor to the experiment
image_size = 128 # Maximum resolution of the training dataset
c = 3 # Number of channels of the dataset
train_funknn = True # Train or just reload to test
restore_funknn = False
training_mode = 'factor' # Training modes for funknn: {conitinuous, factor, single}
ood_analysis = True # Evaluating the performance of model over out of distribution data (Lsun-bedroom)

# Evaluation arguements
max_scale = 4 # Maximum scale to generate in test time (2 or 4 or 8) (=<8 for celeba-hq and 2 for other datasets)
recursive = True # Recursive image reconstructions (Use just for factor training mode)
sample_number = 25 # Number of samples in evaluation


# Datasets paths:
ood_path = 'datasets/lsun_bedroom_val'
train_path = 'datasets/celeba_hq/celeba_hq_1024_train/'
test_path = 'datasets/celeba_hq/celeba_hq_1024_test/'
