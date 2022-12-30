import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import imageio
from funknn_model import grid_sample_customized
from funknn_model import FunkNN
from autoencoder_model import autoencoder, encoder, decoder
from flow_model import real_nvp
from datasets import *
import odl
from ops.ODLHelper import OperatorFunction
import config_IP_solver as config
import config_generative

class temp_var(torch.nn.Module):
	# Learnable latent code object

    def __init__(self, x_init):
        super(temp_var, self).__init__()
        self.x = torch.nn.Parameter(x_init)



class ParallelBeamGeometryOp(object):
	"""Creates an `img_size` mesh parallel geometry tomography operator."""
	def __init__(self, img_size, num_angles, angle_max=np.pi):
		self.img_size = img_size
		self.num_angles = num_angles
		self.reco_space = odl.uniform_discr(
		min_pt=[-1,-1], 
		max_pt=[1,1], 
		shape=[img_size,img_size],
		dtype='float32'
		)

		self.angle_partition = odl.uniform_partition(-angle_max, angle_max, num_angles)    
		self.detector_partition = odl.tomo.Parallel2dGeometry(odl.uniform_partition(0, np.pi, 10), odl.uniform_partition(-1, 1, img_size)).det_partition
		self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition) 

		self.op = odl.tomo.RayTransform(
		self.reco_space,
		self.geometry,
		impl='astra_cuda')

		self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)

	def __call__(self, x):
		return OperatorFunction.apply(self.op, x)

	def pinv(self, y):
		return OperatorFunction.apply(self.fbp, y)


def add_noise(im,SNR):
	perturbation = torch.randn_like(im)
	original_shape = perturbation.shape
	im = im.flatten()
	perturbation = perturbation.flatten()
	truth_norm = torch.linalg.norm(im)
	perturbation_norm = torch.linalg.norm(perturbation)
	k = 1 / ((perturbation_norm / truth_norm)*(10**(SNR/20)))
	perturbation = k*perturbation
	return torch.reshape(im + perturbation, original_shape)

def SNR(x, xhat):
	"""Returns SNR of xhat wrt to gt image x."""

	diff = x - xhat
	return -20*np.log10(np.linalg.norm(diff)/ np.linalg.norm(x))


def inverting_derivatives(image_size, exp_path, funknn, aeder, flow, sparse_derivatives = True):
	'''Reconstructing an image from its spatial derivatives using a generative model as a prior'''

	n_steps = 20000 # Number of iterations
	lr_z = 1e-2 # Learnng rate of the optimizer over the latent code z
	lr_ae = 5e-6 # Learnng rate of the optimizer over the autoencoder weights
	batch_pixels = 10000 # Number of pixels being optimized in one iteration
	lam_z = 0 # Coefficient of likelihhod regularizer (set to zero as we start from z = 0)
	lam_g = 1 # Coefficient of total variation regularizer
	sample_number = 0 # Number of sample in test set to be used
	ae_thresh = 2000 # After this threshold, z fixed and start optimization over autoencoder weights
	pde_desc = 'default' # A short optional discription of your experiment
	derivative_loss = F.mse_loss # or F.l1_loss
	c = 3

	device = funknn.linear1.weight.data.device

	image_path_pde = os.path.join(exp_path, 'PDE_image:' + str(sample_number) + \
		 '_sparse:' + str(sparse_derivatives)) + '_resolution:' + str(image_size) + '_' + pde_desc
	if not os.path.exists(image_path_pde):
		os.mkdir(image_path_pde)

	
	z_init = flow.q0(1)[0]
	mean_flow = flow.q0.loc
	z_init = z_init * 0.0 + mean_flow # Mean of the Gaussian as initialization

	test_dataset = Dataset_loader(dataset = 'test' ,size = (image_size,image_size), c = c, quantize = False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25, num_workers=8)
	image = next(iter(test_loader)).to(device)[sample_number:sample_number + 1]
	c = image.shape[2]
	image = image.reshape(-1, image_size, image_size, c).permute(0,3,1,2)

	# Saving the ground truth image
	image_np = image.permute(0,2,3,1).detach().cpu().numpy()
	image_write = image_np * 255.0
	image_write = image_write[0].clip(0, 255).astype(np.uint8)
	imageio.imwrite(os.path.join(image_path_pde, 'image_gt.png'), image_write)
	
	# Defining the coordinates for the ground truth derivative computations
	t_coords = get_mgrid(image_size).reshape(-1, 2)
	t_coords = torch.unsqueeze(t_coords, dim = 0)
	t_coords = t_coords.clone().detach().requires_grad_(True).to(device)
	t_coords_grad = t_coords.reshape((1,image_size,image_size,2))
	t_coords_grad = 2 * torch.flip(t_coords_grad , dims = [3])
	img_tmp = grid_sample_customized(image, t_coords_grad, pad = 'reflect')
	img_tmp = img_tmp.permute(0,2,3,1).reshape(1 , -1 , c)
	y = gradient(img_tmp, t_coords, grad_outputs=None).detach() # Ground truth derivatives
	
	# Saving the ground truth derivatives
	y_np = torch.sqrt(torch.sum(torch.pow(y,2) , axis = 2))
	y_np = y_np.cpu().numpy()
	y_np = np.reshape(y_np, [-1, image_size, image_size, 1])
	y_write = y_np[0,:,:,0]
	plt.imsave(os.path.join(image_path_pde, 'derivatives_gt.png'), y_write, cmap = 'gray')

	# Just for sprase derivatives
	if sparse_derivatives:
		max_y = torch.max(torch.abs(y),2)[0].detach().cpu().numpy()
		q = np.quantile(max_y,0.8,axis=1,keepdims=True)
		pixels = np.where(max_y>=q)[1]
		sigma_noise = q[0,0]/6*0
		y = y + torch.randn_like(y)*sigma_noise


	# defining a learnable variable for the latent code of the flow model
	z_var = temp_var(z_init).to(device)
	optimizer = torch.optim.Adam(z_var.parameters(), lr= lr_z)
	optimizer_cnn = torch.optim.Adam(aeder.parameters(), lr= lr_ae) # Optimizer over autoencoder weights

	image_vector = image.permute(0,2,3,1).reshape(1 , -1 , c).detach() # GT image as a vector
	psnr_rescale_whole = 0
	with tqdm(total=n_steps) as pbar:
		for i in range(n_steps):
			
			coords = get_mgrid(image_size).reshape(-1, 2)
			coords = torch.unsqueeze(coords, dim = 0).to(device)
			coords = coords.clone().detach().requires_grad_(True) # Dense grid
			if not sparse_derivatives:
				# A random batch of sampled coordinates
				pixels = np.random.randint(low = 0, high = image_size**2, size = batch_pixels)

			batch_coords = coords[:,pixels]
			batch_y = y[:,pixels]
			batch_image_vector = image_vector[:,pixels]

			if i > ae_thresh:
				optimizer_cnn.zero_grad()
			else:
				optimizer.zero_grad()

			z_tilde = flow.sample_me(z_var.x) # Sampling from flow model
			img = aeder.decoder(z_tilde) # Passing through the decoder
			batch_image_vector_hat = funknn(batch_coords , img) # Passing through funknn
			batch_y_hat = gradient(batch_image_vector_hat, batch_coords)
			reg_g = lam_g*torch.mean(torch.norm(batch_y_hat,p=1,dim=2)) # Total variation regularization
			reg_z = lam_z * flow.forward_kld(z_tilde) # Maximum likelihood regularization
			loss = derivative_loss(batch_y, batch_y_hat) + reg_z + reg_g

			if i > ae_thresh :
				loss.backward()
				optimizer_cnn.step()
			else:
				loss.backward()
				optimizer.step()

			batch_image_vector_hat = batch_image_vector_hat.detach().cpu().numpy()
			batch_image_vector = batch_image_vector.detach().cpu().numpy()
			psnr = PSNR(batch_image_vector, batch_image_vector_hat)
			psnr_rescale = PSNR_rescale(batch_image_vector, batch_image_vector_hat)[0]
			pbar.set_description('Loss: {:.2f}| psnr: {:.2f} | psnr_rescale: {:.2f} | | psnr_rescale_whole: {:.2f}'.format(loss, psnr, psnr_rescale, psnr_rescale_whole))
			pbar.update(1)

			if i % 100 == 0:
				# Visualization
				# Reconstructed image
				coords = get_mgrid(image_size).reshape(-1, 2)
				coords = torch.unsqueeze(coords, dim = 0)			    
				recon_np = batch_sampling(img, coords,c, funknn)
				recon_np = np.reshape(recon_np, [-1, image_size, image_size, c])
				recon_np = recon_np.reshape(1,-1)
				image_np = image_np.reshape(1,-1)
				psnr_rescale_whole, weights = PSNR_rescale(image_np, recon_np)
				recon_np = weights[0]*recon_np+weights[1]
				# recon_np = recon_np - recon_np.min(axis = -1, keepdims = True)
				# recon_np  = recon_np/recon_np.max(axis = -1, keepdims = True)
				recon_np = recon_np.reshape(-1, image_size, image_size, c) * 255.0
				recon_write = recon_np[0].clip(0, 255).astype(np.uint8)
				imageio.imwrite(os.path.join(image_path_pde, '%d_recon.png' % (i,)), recon_write)

				# Reconstruced gradients:
				coords = get_mgrid(image_size).reshape(-1, 2)
				coords = torch.unsqueeze(coords, dim = 0)
				coords = coords.clone().detach().requires_grad_(True)		    
				recon_np = batch_grad(img, coords,c, funknn)
				recon_np = np.reshape(recon_np, [-1, image_size, image_size, 1])
				recon_write = recon_np[0,:,:,0]
				plt.imsave(os.path.join(image_path_pde, '%d_recon_grad.png' % (i,)),
							recon_write, cmap = 'gray')

				with open(os.path.join(image_path_pde, 'results.txt'), 'a') as file:
					file.write('iter: {:.0f}| Loss: {:.2f}| psnr: {:.2f}  | psnr_rescale_whole: {:.2f}'.format(i,loss, psnr, psnr_rescale, psnr_rescale_whole))
					file.write('\n')




def limited_view_CT(image_size, exp_path, funknn, aeder, flow):

	lr_z = 1e-1
	sample_number = 0
	weight_data_fidelity = 1
	n_steps = 5000
	SNR_target = 30
	angle_max = 70
	mse_loss = F.mse_loss
	run_desc = 'default'


	image_path = os.path.join(exp_path, 'CT_image:' + str(sample_number) + \
		'_resolution:' + str(image_size) + '_' + run_desc)
	if not os.path.exists(image_path):
		os.mkdir(image_path)


	op_high_res = ParallelBeamGeometryOp(image_size,num_angles=200,angle_max=70*np.pi/180)

	z_init = flow.q0(1)[0]
	mean_flow = flow.q0.loc
	z_init = z_init * 0.0 + mean_flow
	z_var = temp_var(z_init).to(device)
	optimizer = torch.optim.Adam(z_var.parameters(), lr=lr_z)

	test_dataset = Dataset_loader(dataset = 'test' ,size = (image_size,image_size), c = c, quantize = False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25, num_workers=8)
	image = next(iter(test_loader)).to(device)[sample_number:sample_number + 1,:,0:1]
	image = image.reshape(-1, image_size, image_size, 1).permute(0,3,1,2)

	# Saving gt
	im_save = np.squeeze(image.detach().cpu().numpy())*255.
	im_save = im_save.clip(0, 255).astype(np.uint8)
	imageio.imwrite(os.path.join(image_path, 'im' + 'gt.png'),
				im_save)

	# CT forward operator
	op_high_res = ParallelBeamGeometryOp(image_size,num_angles=200,angle_max=angle_max*np.pi/180)
	y_obs = op_high_res(image[:,0])
	y_obs = add_noise(y_obs,SNR_target)

	# Saving bp
	im_noise = op_high_res.pinv(y_obs)
	im_save = np.squeeze(im_noise.detach().cpu().numpy())*255.
	im_save = im_save.clip(0, 255).astype(np.uint8)
	imageio.imwrite(os.path.join(image_path, 'bp.png'),
				im_save) 

	with tqdm(total=n_steps) as pbar:
		for i in tqdm(range(n_steps)):
			coords = get_mgrid(image_size).reshape(-1, 2)
			coords = torch.unsqueeze(coords, dim = 0)
			coords = coords.clone().detach().requires_grad_(True)

			optimizer.zero_grad()
			img = aeder.decoder(flow.sample_me(z_var.x))
			est = funknn(coords , img)
			im_est = torch.reshape(est[:,:,0:1],(1,image_size,image_size))
			y_hat = op_high_res(im_est)
			
			l_mse = weight_data_fidelity*mse_loss(y_obs, y_hat)
			loss = l_mse
			loss.backward()
			optimizer.step()

			psnr = PSNR(image.detach().cpu().numpy(),im_est.detach().cpu().numpy())
			pbar.set_description('Loss: {:.2f}| psnr: {:.2f}'.format(loss, psnr))
			pbar.update(1)

			if i % 100 == 0:
				# Visualization
				# FunkNN Reconstruction
				im_save = np.squeeze(im_est.detach().cpu().numpy())*255.
				im_save = im_save.clip(0, 255).astype(np.uint8)
				imageio.imwrite(os.path.join(image_path, '%d_recon.png' % (i,)),
							im_save) 

				# AE Reconstruction
				im_save = np.squeeze(img.detach().cpu().numpy())*255.
				im_save = im_save.clip(0, 255).astype(np.uint8)
				imageio.imwrite(os.path.join(image_path, '%d_recon_ae.png' % (i,)),
							im_save)

				with open(os.path.join(image_path, 'results.txt'), 'a') as file:
					file.write('iter: {:.0f}| Loss: {:.2f}| psnr: {:.2f}'.format(i,loss, psnr))
					file.write('\n')




if __name__ == '__main__':

	gpu_num = config.gpu_num
	image_size = config.image_size # Working resolution for solving inverse problems
	problem = config.problem # inverse problem:{CT, PDE}
	sparse_derivatives = config.sparse_derivatives # Sparse derivative option, just for PDE problem
	funknn_path = config.funknn_path  # Trained Funknn  folder
	autoencoder_path = config.autoencoder_path # Trained generative autoencoder folder
	exp_desc = config.exp_desc

	# Modify these variables based on the models you are loading
	ae_image_size = config_generative.image_size # Image resolution of autoencoder
	latent_dim = config_generative.latent_dim # Latent_dim of atutoencoder
	c = config_generative.c # Channel size of dataset (RGB ot grayscale)
	flow_depth = config_generative.flow_depth
	dataset = config_generative.dataset

	# Print the experiment setup:
	print('Experiment setup:')
	print('---> problem: {}'.format(problem))
	print('---> image size: {}'.format(image_size))

	enable_cuda = True
	device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')

	inverse_problem_folder = 'Inverse_problems/'
	if os.path.exists(inverse_problem_folder) == False:
		os.mkdir(inverse_problem_folder)

	# Experiment path: The experiments runned using these versions of Funknn and generator
	exp_path = inverse_problem_folder  + dataset + '_' + exp_desc

	if os.path.exists(exp_path) == False:
		os.mkdir(exp_path)

	#  Loading Funknn
	model = FunkNN(c=c).to(device)
	checkpoint_funknn_path = os.path.join(funknn_path, 'funknn.pt')
	if os.path.exists(checkpoint_funknn_path):	
		checkpoint_funknn = torch.load(checkpoint_funknn_path)
		model.load_state_dict(checkpoint_funknn['model_state_dict'])
		print('funknn is restored...')

	# Loading Autoencoder
	enc = encoder(latent_dim = latent_dim, in_res = ae_image_size , c = c).to(device)
	dec = decoder(latent_dim = latent_dim, in_res = ae_image_size , c = c).to(device)
	aeder = autoencoder(encoder = enc , decoder = dec).to(device)
	checkpoint_autoencoder_path = os.path.join(autoencoder_path, 'autoencoder.pt')
	if os.path.exists(checkpoint_autoencoder_path):
		checkpoint_autoencoder = torch.load(checkpoint_autoencoder_path)
		aeder.load_state_dict(checkpoint_autoencoder['model_state_dict'])
		print('Autoencoder is restored...')


	# Loading Flow
	nfm = real_nvp(latent_dim = latent_dim, K = flow_depth).to(device)
	checkpoint_flow_path = os.path.join(autoencoder_path, 'flow.pt')
	if os.path.exists(checkpoint_flow_path):
		checkpoint_flow = torch.load(checkpoint_flow_path)
		nfm.load_state_dict(checkpoint_flow['model_state_dict'])
		print('Flow is restored...')

	if problem == 'PDE':
		inverting_derivatives(image_size, exp_path, funknn = model, aeder = aeder, 
		flow = nfm,sparse_derivatives = sparse_derivatives)
	elif problem == 'CT':
		limited_view_CT(image_size, exp_path, funknn = model, aeder = aeder, flow = nfm)