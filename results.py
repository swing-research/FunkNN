import numpy as np
import torch
import torch.nn.functional as F
import os
import imageio
from utils import *
import config_funknn as config
import matplotlib.pyplot as plt


def evaluator(ep, subset, data_loader, model, exp_path):

    samples_folder = os.path.join(exp_path, 'Results')
    if not os.path.exists(samples_folder):
        os.mkdir(samples_folder)
    image_path_reconstructions = os.path.join(
        samples_folder, 'Reconstructions')

    if not os.path.exists(image_path_reconstructions):
        os.mkdir(image_path_reconstructions)

    max_scale = config.max_scale
    recursive = config.recursive
    sample_number = config.sample_number
    image_size = config.image_size
    c = config.c

    if subset == 'ood':
        max_scale = 2
        recursive = False

    device = model.ws1.device
    num_samples_write = sample_number if sample_number < 26 else 25
    ngrid = int(np.sqrt(num_samples_write))
    num_samples_write = int(ngrid **2)

    images_k = next(iter(data_loader)).to(device)[:sample_number]
    images_k = images_k.reshape(-1, max_scale*image_size, max_scale*image_size, c).permute(0,3,1,2)
    images = F.interpolate(images_k, size = image_size, antialias = True, mode = 'bilinear')

    scales = [i+1 for i in range(int(np.log2(max_scale)))]
    scales = np.power(2, scales)

    print('Evaluation over {} set:'.format(subset))
    with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
        file.write('Evaluation over {} set:'.format(subset))
        file.write('\n')

    if recursive == True:

        images_down = images

        for i in range(len(scales)):
            # Recuirsive image generation starting from factor 2 well-suited for factor training mode
            res = scales[i]*image_size
            # GT:
            images_temp = F.interpolate(images_k, size = res , antialias = True, mode = 'bilinear')
            images_np = images_temp.permute(0, 2, 3, 1).detach().cpu().numpy()
            image_write = images_np[:num_samples_write].reshape(
                ngrid, ngrid,
                res, res,c).swapaxes(1, 2).reshape(ngrid*res, -1, c)*255.0
            image_write = image_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, subset +  '_%d_gt_%d.png' % (ep,scales[i])),
                        image_write)

            # Recon:
            coords = get_mgrid(res).reshape(-1, 2)
            coords = torch.unsqueeze(coords, dim = 0)
            coords = coords.expand(images_k.shape[0] , -1, -1).to(device)
            recon_np = batch_sampling(images_down, coords,c, model)
            recon_np = np.reshape(recon_np, [-1, res, res, c])
            recon_write = recon_np[:num_samples_write].reshape(
                ngrid, ngrid, res, res, c).swapaxes(1, 2).reshape(ngrid*res, -1, c)*255.0
            recon_write = recon_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, subset + '_%d_recursive_FunkNN_%d.png' % (ep,scales[i])),
                        recon_write)

            # Interpolate:
            interpolate = F.interpolate(images, size = res, mode = 'bilinear')
            interpolate_np = interpolate.detach().cpu().numpy().transpose(0,2,3,1)
            interpolate_write = interpolate_np[:num_samples_write].reshape(
                ngrid, ngrid,
                res, res, c).swapaxes(1, 2).reshape(ngrid*res, -1, c)*255.0
            interpolate_write = interpolate_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, subset + '_%d_interpolate_%d.png' % (ep,scales[i])),
                            interpolate_write) # mesh_based_recon

            snr_recon = SNR(images_np, recon_np)
            snr_interpolate = SNR(images_np, interpolate_np)
            recon_np = recon_np.transpose([0,3,1,2])
            images_down = torch.tensor(recon_np, dtype = images_down.dtype).to(device)

            print('SNR_interpolate_recursive_f{}: {:.1f} | SNR_FunkNN_recursive_f{}: {:.1f}'.format(scales[i],
                snr_interpolate, scales[i], snr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('SNR_interpolate_recursive_f{}: {:.1f} | SNR_FunkNN_recursive_f{}: {:.1f} | '.format(scales[i],
                snr_interpolate, scales[i], snr_recon))
                file.write('\n')
                if subset == 'ood':
                    file.write('\n')

    else:
        for i in range(len(scales)):
            # Direct image generation well-suited for single and continuous training modes
            res = scales[i]*image_size
            # GT:
            images_temp = F.interpolate(images_k, size = res , antialias = True, mode = 'bilinear')
            images_np = images_temp.permute(0, 2, 3, 1).detach().cpu().numpy()
            image_write = images_np[:num_samples_write].reshape(
                ngrid, ngrid,
                res, res,c).swapaxes(1, 2).reshape(ngrid*res, -1, c)*255.0
            image_write = image_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, subset +  '_%d_gt_%d.png' % (ep,scales[i])),
                        image_write)

            # Recon:
            coords = get_mgrid(res).reshape(-1, 2)
            coords = torch.unsqueeze(coords, dim = 0)
            coords = coords.expand(images_k.shape[0] , -1, -1).to(device)
            recon_np = batch_sampling(images, coords,c, model)
            recon_np = np.reshape(recon_np, [-1, res, res, c])
            recon_write = recon_np[:num_samples_write].reshape(
                ngrid, ngrid, res, res, c).swapaxes(1, 2).reshape(ngrid*res, -1, c)*255.0
            recon_write = recon_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, subset + '_%d_FunkNN_%d.png' % (ep,scales[i])),
                        recon_write)

            # Interpolate:
            interpolate = F.interpolate(images, size = res, mode = 'bilinear')
            interpolate_np = interpolate.detach().cpu().numpy().transpose(0,2,3,1)
            interpolate_write = interpolate_np[:num_samples_write].reshape(
                ngrid, ngrid,
                res, res, c).swapaxes(1, 2).reshape(ngrid*res, -1, c)*255.0
            interpolate_write = interpolate_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, subset + '_%d_interpolate_%d.png' % (ep,scales[i])),
                            interpolate_write) # mesh_based_recon

            snr_recon = SNR(images_np, recon_np)
            snr_interpolate = SNR(images_np, interpolate_np)

            print('SNR_interpolate_f{}: {:.1f} | SNR_FunkNN_f{}: {:.1f}'.format(scales[i],
                snr_interpolate, scales[i], snr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('SNR_interpolate_f{}: {:.1f} | SNR_FunkNN_f{}: {:.1f} | '.format(scales[i],
                snr_interpolate, scales[i], snr_recon))
                file.write('\n')
                if subset == 'ood':
                    file.write('\n')




    if config.derivatives_evaluation:
        # Gradients:
        coords_2k = get_mgrid(2*image_size).reshape(-1, 2)
        coords_2k = torch.unsqueeze(coords_2k, dim = 0)
        coords_2k = coords_2k.expand(images_k.shape[0] , -1, -1).to(device)
        coords_2k = coords_2k.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        funknn_grad = batch_grad(images, coords_2k,c, model)
        funknn_grad = torch.tensor(funknn_grad, dtype=coords.dtype)
        funknn_grad = torch.norm(funknn_grad, dim = 2).cpu().detach().numpy()
        funknn_grad = np.reshape(funknn_grad, [-1, 2*image_size, 2*image_size,1])
        funknn_grad_write = funknn_grad[:sample_number, :, :].reshape(
            ngrid, ngrid,
            2*image_size, 2*image_size, 1).swapaxes(1, 2).reshape(ngrid*2*image_size, -1, 1)*255.0
        plt.imsave(os.path.join(image_path_reconstructions, subset + '_%d_grad_funknn.png' % (ep,)),
                        funknn_grad_write[:,:,0], cmap='seismic')


        coords_2k = get_mgrid(2*image_size).reshape(-1, 2)
        coords_2k = torch.unsqueeze(coords_2k, dim = 0)
        coords_2k = coords_2k.expand(images_k.shape[0] , -1, -1).to(device)
        coords_2k = coords_2k.clone().detach().requires_grad_(True) 
        funknn = torch.tensor(batch_sampling(images, coords_2k,c, model), dtype = coords_2k.dtype).to(device)
        funknn_mat = funknn.reshape([-1, 2*image_size, 2*image_size, c]).permute(0,3,1,2)
        true_grad = image_derivative(funknn_mat , c)[1]
        true_grad = true_grad.permute(0,2,3,1).detach().cpu().numpy()
        true_grad_write = true_grad[:sample_number, :, :].reshape(
            ngrid, ngrid,
            2*image_size, 2*image_size, 1).swapaxes(1, 2).reshape(ngrid*2*image_size, -1, 1)*255.0

        plt.imsave(os.path.join(image_path_reconstructions, subset + '_%d_grad_finite_diff.png' % (ep,)),
                        true_grad_write[:,:,0], cmap='seismic')

        coords_2k = get_mgrid(2*image_size).reshape(-1, 2)
        coords_2k = torch.unsqueeze(coords_2k, dim = 0)
        coords_2k = coords_2k.expand(images_k.shape[0] , -1, -1).to(device)
        coords_2k = coords_2k.clone().detach().requires_grad_(True).to(device)
        coords_2k_grad = coords_2k.reshape((-1,2*image_size,2*image_size,2))
        coords_2k_grad = 2 * torch.flip(coords_2k_grad , dims = [3])
        img_tmp = model.grid_sample_customized(images, coords_2k_grad, mode = config.interpolation_kernel)
        img_tmp = img_tmp.permute(0,2,3,1).reshape(1 , -1 , c)
        st_grad = gradient(img_tmp, coords_2k, grad_outputs=None)
        st_grad = torch.norm(st_grad, dim = 2).cpu().detach().numpy()
        st_grad = np.reshape(st_grad, [-1, 2*image_size, 2*image_size, 1])
        st_grad_write = st_grad[:sample_number, :, :].reshape(
            ngrid, ngrid,
            2*image_size, 2*image_size, 1).swapaxes(1, 2).reshape(ngrid*2*image_size, -1, 1)*255.0

        plt.imsave(os.path.join(image_path_reconstructions, subset + '_%d_grad_ST.png' % (ep,)),
                        st_grad_write[:,:,0], cmap='seismic')

 

    ############################################################################################
    # Laplacian:
        shift = 4
        coords_2k = get_mgrid(2*image_size).reshape(-1, 2)
        coords_2k = torch.unsqueeze(coords_2k, dim = 0)
        coords_2k = coords_2k.expand(images_k.shape[0] , -1, -1).to(device)
        coords_2k = coords_2k.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        funknn_laplace= batch_laplace(images, coords_2k,c, model)
        funknn_laplace = np.reshape(funknn_laplace, [-1, 2*image_size, 2*image_size,1])

        funknn_laplace_write = funknn_laplace[:images_k.shape[0], shift:2*image_size-shift, shift:2*image_size-shift].reshape(
            ngrid, ngrid,
            2*image_size -2*shift, 2*image_size-2*shift, 1).swapaxes(1, 2).reshape(ngrid*(2*image_size-2*shift), -1, 1)

        plt.imsave(os.path.join(image_path_reconstructions, subset + '_%d_laplace_funknn.png' % (ep,)),
                        funknn_laplace_write[:,:,0], cmap='seismic')
 
        coords_2k = get_mgrid(2*image_size).reshape(-1, 2)
        coords_2k = torch.unsqueeze(coords_2k, dim = 0)
        coords_2k = coords_2k.expand(images_k.shape[0] , -1, -1).to(device)
        coords_2k = coords_2k.clone().detach().requires_grad_(True).to(device)
        coords_2k_grad = coords_2k.reshape((-1,2*image_size,2*image_size,2))
        coords_2k_grad = 2 * torch.flip(coords_2k_grad , dims = [3])
        img_tmp = model.grid_sample_customized(images, coords_2k_grad, mode = config.interpolation_kernel)
        img_tmp = img_tmp.permute(0,2,3,1).reshape(1 , -1 , c)
        st_laplace = laplace(img_tmp, coords_2k).detach() # Ground truth derivatives
        st_laplace = np.reshape(st_laplace.cpu().numpy(), [-1, 2*image_size, 2*image_size, 1])
        st_laplace_write = st_laplace[:images_k.shape[0], shift:2*image_size-shift, shift:2*image_size-shift].reshape(
            ngrid, ngrid,
            2*image_size -2*shift, 2*image_size-2*shift, 1).swapaxes(1, 2).reshape(ngrid*(2*image_size-2*shift), -1, 1)

        plt.imsave(os.path.join(image_path_reconstructions, subset + '_%d_laplace_ST.png' % (ep,)),
                        st_laplace_write[:,:,0], cmap='seismic')