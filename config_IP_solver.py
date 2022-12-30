gpu_num = 0
image_size = 256 # Working resolution for solving inverse problems
problem = 'PDE' # inverse problem:{CT, PDE}
sparse_derivatives = False # Sparse derivative option, just for PDE problem
funknn_path = 'experiments/' +  'funknn_celeba-hq_128_factor_default' # Trained Funknn  folder
autoencoder_path = 'experiments/' + 'generator_celeba-hq_5_256_128_default' # Trained generative autoencoder folder
exp_desc = 'default' # A note to indicate which version of funknn and generative autoencoder are combined