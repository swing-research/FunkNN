import numpy as np
import torch
import torch.nn.functional as F
import os
import imageio
from utils import *
import config_funknn as config


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
            imageio.imwrite(os.path.join(image_path_reconstructions, subset + '_%d_recursive_recon_%d.png' % (ep,scales[i])),
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

            print('SNR_interpolate_recursive_f{}: {:.1f} | SNR_recon_recursive_f{}: {:.1f}'.format(scales[i],
                snr_interpolate, scales[i], snr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('SNR_interpolate_recursive_f{}: {:.1f} | SNR_recon_recursive_f{}: {:.1f} | '.format(scales[i],
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
            imageio.imwrite(os.path.join(image_path_reconstructions, subset + '_%d_recon_%d.png' % (ep,scales[i])),
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

            print('SNR_interpolate_f{}: {:.1f} | SNR_recon_f{}: {:.1f}'.format(scales[i],
                snr_interpolate, scales[i], snr_recon))

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                file.write('SNR_interpolate_f{}: {:.1f} | SNR_recon_f{}: {:.1f} | '.format(scales[i],
                snr_interpolate, scales[i], snr_recon))
                file.write('\n')
                if subset == 'ood':
                    file.write('\n')