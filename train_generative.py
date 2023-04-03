import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
from torch.optim import Adam
import os
import imageio
import matplotlib.pyplot as plt
from autoencoder_model import autoencoder, encoder, decoder
from flow_model import real_nvp
from utils import *
from datasets import *
from laplacian_loss import LaplacianPyramidLoss
import config_generative as config

torch.manual_seed(0)
np.random.seed(0)

epochs_flow = config.epochs_flow
epochs_aeder = config.epochs_aeder
flow_depth = config.flow_depth
latent_dim = config.latent_dim
batch_size = config.batch_size
dataset = config.dataset
gpu_num = config.gpu_num
exp_desc = config.exp_desc
image_size = config.image_size
c = config.c
train_aeder = config.train_aeder
train_flow = config.train_flow
restore_flow = config.restore_flow
restore_aeder = config.restore_aeder

enable_cuda = True
device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() and enable_cuda else 'cpu')


all_experiments = 'experiments/'
if os.path.exists(all_experiments) == False:
    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + 'generator_' + dataset + '_' \
    + str(flow_depth) + '_' + str(latent_dim) + '_' + str(image_size) + '_' + exp_desc

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


learning_rate = 1e-4
step_size = 50
gamma = 0.5
lam = 0.01

# Print the experiment setup:
print('Experiment setup:')
print('---> epochs_aeder: {}'.format(epochs_aeder))
print('---> epochs_flow: {}'.format(epochs_flow))
print('---> flow_depth: {}'.format(flow_depth))
print('---> batch_size: {}'.format(batch_size))
print('---> dataset: {}'.format(dataset))
print('---> Learning rate: {}'.format(learning_rate))
print('---> experiment path: {}'.format(exp_path))
print('---> latent dim: {}'.format(latent_dim))
print('---> image size: {}'.format(image_size))


# Dataset:
train_dataset = Dataset_loader(dataset = 'train' ,size = (image_size,image_size), c = c, quantize = False)
test_dataset = Dataset_loader(dataset = 'test' ,size = (image_size,image_size), c = c, quantize = False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=40, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25, num_workers=8)

ntrain = len(train_loader.dataset)
n_test = len(test_loader.dataset)
print('---> Number of training, test samples: {}, {}'.format(ntrain,n_test))
plot_per_num_epoch = 1 if ntrain > 20000 else 20000//ntrain

# Loss
dum_samples = next(iter(test_loader)).to(device)
mse_l = F.mse_loss
pyramid_l = LaplacianPyramidLoss(max_levels=3, channels=c, kernel_size=5,
    sigma=1, device=device, dtype=dum_samples.dtype)
vgg =Vgg16().to(device)
for param in vgg.parameters():
    param.requires_grad = False

# 1. Training Autoencoder:
enc = encoder(latent_dim = latent_dim, in_res = image_size , c = c).to(device)
dec = decoder(latent_dim = latent_dim, in_res = image_size , c = c).to(device)
aeder = autoencoder(encoder = enc , decoder = dec).to(device)

num_param_aeder= count_parameters(aeder)
print('---> Number of trainable parameters of Autoencoder: {}'.format(num_param_aeder))

optimizer_aeder = Adam(aeder.parameters(), lr=learning_rate)
scheduler_aeder = torch.optim.lr_scheduler.StepLR(optimizer_aeder, step_size=step_size, gamma=gamma)

checkpoint_autoencoder_path = os.path.join(exp_path, 'autoencoder.pt')
if os.path.exists(checkpoint_autoencoder_path) and restore_aeder == True:
    checkpoint_autoencoder = torch.load(checkpoint_autoencoder_path)
    aeder.load_state_dict(checkpoint_autoencoder['model_state_dict'])
    optimizer_aeder.load_state_dict(checkpoint_autoencoder['optimizer_state_dict'])
    print('Autoencoder is restored...')


if train_aeder:

    if plot_per_num_epoch == -1:
        plot_per_num_epoch = epochs_aeder + 1 # only plot in the last epoch
    
    loss_ae_plot = np.zeros([epochs_aeder])
    for ep in range(epochs_aeder):
        aeder.train()
        t1 = default_timer()
        loss_ae_epoch = 0

        # Training 100 rpochs over style and then over combined loss of style and mse
        loss_type = 'style' if ep < 100 else 'style_mse'
        for image in train_loader:
            
            batch_size = image.shape[0]
            image = image.to(device)
            
            optimizer_aeder.zero_grad()
            image_mat = image.reshape(-1, image_size, image_size, c).permute(0,3,1,2)

            embed = aeder.encoder(image_mat)
            image_recon = aeder.decoder(embed)

            recon_loss = aeder_loss(image_mat, image_recon, loss_type = loss_type,
                pyramid_l = pyramid_l, mse_l = mse_l, vgg = vgg)
            regularization = mse_l(embed, torch.zeros(embed.shape).to(device))
            ae_loss = recon_loss + lam * regularization

            ae_loss.backward()
            optimizer_aeder.step()
            loss_ae_epoch += ae_loss.item()

        scheduler_aeder.step()
        t2 = default_timer()
        loss_ae_epoch/= ntrain
        loss_ae_plot[ep] = loss_ae_epoch
        
        plt.plot(np.arange(epochs_aeder)[:ep], loss_ae_plot[:ep], 'o-', linewidth=2)
        plt.title('AE_loss')
        plt.xlabel('epoch')
        plt.ylabel('MSE loss')

        plt.savefig(os.path.join(exp_path, 'Autoencoder_loss.jpg'))
        np.save(os.path.join(exp_path, 'Autoencoder_loss.npy'), loss_ae_plot[:ep])
        plt.close()
        
        torch.save({
                    'model_state_dict': aeder.state_dict(),
                    'optimizer_state_dict': optimizer_aeder.state_dict()
                    }, checkpoint_autoencoder_path)


        samples_folder = os.path.join(exp_path, 'Results')
        if not os.path.exists(samples_folder):
            os.mkdir(samples_folder)
        image_path_reconstructions = os.path.join(
            samples_folder, 'Reconstructions_aeder')
    
        if not os.path.exists(image_path_reconstructions):
            os.mkdir(image_path_reconstructions)
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_aeder:
            sample_number = 25
            ngrid = int(np.sqrt(sample_number))

            test_images = next(iter(test_loader)).to(device)[:sample_number]
            test_images = test_images.reshape(-1, image_size, image_size, c).permute(0,3,1,2)

            image_np = test_images.permute(0,2,3,1).detach().cpu().numpy()
            image_write = image_np[:sample_number].reshape(
                ngrid, ngrid,
                image_size, image_size,c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0
            image_write = image_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, '%d_gt.png' % (ep,)),image_write)
            
            embed = aeder.encoder(test_images)
            image_recon = aeder.decoder(embed)
            image_recon_np = image_recon.detach().cpu().numpy().transpose(0,2,3,1)
            image_recon_write = image_recon_np[:sample_number].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0

            image_recon_write = image_recon_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(image_path_reconstructions, '%d_aeder_recon.png' % (ep,)),
                            image_recon_write)
            
            snr_aeder = SNR(image_np , image_recon_np)
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                        file.write('ep: %03d/%03d | time: %.0f | aeder_loss %.4f | SNR_aeder  %.4f' %(ep, epochs_aeder,t2-t1,
                            loss_ae_epoch, snr_aeder))
                        file.write('\n')
            print('ep: %03d/%03d | time: %.0f | aeder_loss %.4f | SNR_aeder  %.4f' %(ep, epochs_aeder,t2-t1,
                            loss_ae_epoch, snr_aeder))
        

# Training the flow model
nfm = real_nvp(latent_dim = latent_dim, K = flow_depth)
nfm = nfm.to(device)
num_param_nfm = count_parameters(nfm)
print('Number of trainable parametrs of flow: {}'.format(num_param_nfm))

loss_hist = np.array([])
optimizer_flow = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=step_size, gamma=gamma)

# Initialize ActNorm
batch_img = next(iter(train_loader)).to(device)
batch_img = batch_img.reshape(-1, image_size, image_size, c).permute(0,3,1,2)
dummy_samples = aeder.encoder(batch_img)
# dummy_samples = model.reference_latents(torch.tensor(0).to(device))
dummy_samples = dummy_samples.view(-1, latent_dim)
# dummy_samples = torch.tensor(dummy_samples).float().to(device)
likelihood = nfm.log_prob(dummy_samples)

checkpoint_flow_path = os.path.join(exp_path, 'flow.pt')
if os.path.exists(checkpoint_flow_path) and restore_flow == True:
    checkpoint_flow = torch.load(checkpoint_flow_path)
    nfm.load_state_dict(checkpoint_flow['model_state_dict'])
    optimizer_flow.load_state_dict(checkpoint_flow['optimizer_state_dict'])
    print('Flow model is restored...')

if train_flow:
    
    for ep in range(epochs_flow):

        nfm.train()
        t1 = default_timer()
        loss_flow_epoch = 0
        for image in train_loader:
            optimizer_flow.zero_grad()
            image = image.to(device)
            image = image.reshape(-1, image_size, image_size, c).permute(0,3,1,2)

            x = aeder.encoder(image)
            # Compute loss
            loss_flow = nfm.forward_kld(x)
            
            if ~(torch.isnan(loss_flow) | torch.isinf(loss_flow)):
                loss_flow.backward()
                optimizer_flow.step()
            
            # Make layers Lipschitz continuous
            # nf.utils.update_lipschitz(nfm, 5)
            loss_flow_epoch += loss_flow.item()
            # Log loss
            loss_hist = np.append(loss_hist, loss_flow.to('cpu').data.numpy())
        
        scheduler_flow.step()
        t2 = default_timer()
        loss_flow_epoch /= ntrain
        
        torch.save({
                    'model_state_dict': nfm.state_dict(),
                    'optimizer_state_dict': optimizer_flow.state_dict()
                    }, checkpoint_flow_path)
        
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_flow:
            samples_folder = os.path.join(exp_path, 'Results')
            if not os.path.exists(samples_folder):
                os.mkdir(samples_folder)
            image_path_generated = os.path.join(
                samples_folder, 'generated')
        
            if not os.path.exists(image_path_generated):
                os.mkdir(image_path_generated)
            sample_number = 25
            ngrid = int(np.sqrt(sample_number))
            
            generated_embed, _ = nfm.sample(torch.tensor(sample_number).to(device))
            
            generated_samples = aeder.decoder(generated_embed)
            generated_samples = generated_samples.detach().cpu().numpy().transpose(0,2,3,1)

            generated_samples = generated_samples[:sample_number].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0
            generated_samples = generated_samples.clip(0, 255).astype(np.uint8)
            
              
            imageio.imwrite(os.path.join(image_path_generated, 'epoch %d.png' % (ep,)), generated_samples) # training images
            
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                    file.write('ep: %03d/%03d | time: %.0f | ML_loss %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch))
                    file.write('\n')
    
            print('ep: %03d/%03d | time: %.0f | ML_loss %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch))

