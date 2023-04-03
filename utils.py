import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from skimage.transform import radon, iradon
from scipy import optimize
from skimage.metrics import peak_signal_noise_ratio as psnr


def image_derivative(x , c):
    '''x must be a (b,c*h*w) tensor'''
    
    horiz_derive = np.array([[1, 0, -1],[2, 0, -2],[1,0,-1]], dtype = np.float64)
    horiz_derive = horiz_derive[None,None,:]
    horiz_derive = np.repeat(horiz_derive,c,axis = 1)

    vert_derive = np.array([[1,2,1],[0,0,0], [-1,-2,-1]])
    vert_derive = vert_derive[None,None,:]
    vert_derive = np.repeat(vert_derive,c,axis = 1)

    conv_horiz = torch.nn.Conv2d(1, c, kernel_size=3, stride=1, padding='same', padding_mode = 'replicate',bias=False)
    conv_horiz.weight.data= torch.from_numpy(horiz_derive).float().to(x.device)

    conv_vert = torch.nn.Conv2d(1, c, kernel_size=3, stride=1, padding='same', padding_mode = 'replicate', bias=False)
    conv_vert.weight.data= torch.from_numpy(vert_derive).float().to(x.device)

    G_x = conv_horiz(x)
    G_y = conv_vert(x)
    G = torch.cat((G_x , G_y) , axis = 1)
    G_mag = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))

    return G, G_mag


def PSNR(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_pred[i],
             x_true[i],
             data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]

def SNR(x_true , x_pred):
    '''Calculate SNR of a batch of true and their estimations'''
        
    # x_true = np.reshape(x_true , [np.shape(x_true)[0] , -1])
    # x_pred = np.reshape(x_pred , [np.shape(x_pred)[0] , -1])
    
    snr = 0
    for i in range(x_true.shape[0]):
        Noise = x_true[i] - x_pred[i]
        Noise_power = np.sum(np.square(np.abs(Noise)))
        Signal_power = np.sum(np.square(np.abs(x_true[i])))
        snr += 10*np.log10(Signal_power/Noise_power)
  
    return snr/x_true.shape[0]



def SNR_rescale(x_true , x_pred):
    '''Calculate SNR rescale of a batch of true and their estimations'''
    snr = 0
    for i in range(x_true.shape[0]):

        def func(weights):
            Noise = x_true[i] - (weights[0]*x_pred[i]+weights[1])
            Noise_power = np.sum(np.square(np.abs(Noise)))
            Signal_power = np.sum(np.square(np.abs(x_true[i])))
            SNR = 10*np.log10(np.mean(Signal_power/(Noise_power+1e-12)))
            return SNR
        opt = optimize.minimize(lambda x: -func(x),x0=np.array([1,0]))
        snr += -opt.fun
        weights = opt.x
    return snr/x_true.shape[0]


def PSNR_rescale(x_true , x_pred):
    '''Calculate SNR rescale of a batch of true and their estimations'''
    snr = 0
    for i in range(x_true.shape[0]):
        
        def func(weights):
            x_pred_rescale=  weights[0]*x_pred[i]+weights[1]
            s = psnr(x_pred_rescale,
             x_true[i],
             data_range=x_true[i].max() - x_true[i].min())
            
            return s
        opt = optimize.minimize(lambda x: -func(x),x0=np.array([1,0]))
        snr += -opt.fun
        weights = opt.x
    return snr/x_true.shape[0], weights



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def relative_mse_loss(x_true, x_pred):
    noise_power = np.sqrt(np.sum(np.square(x_true - x_pred) , axis = [1,2,3]))
    signal_power = np.sqrt(np.sum(np.square(x_true) , axis = [1,2,3]))
    
    return np.mean(noise_power/signal_power)


def batch_sampling(image_recon, coords, c, model):
    s = 64
    
    outs = np.zeros([np.shape(coords)[0], np.shape(coords)[1], c])
    for i in range(np.shape(coords)[1]//s):
        
        batch_coords = coords[:,i*s: (i+1)*s]
        out = model(batch_coords, image_recon).detach().cpu().numpy()
        outs[:,i*s: (i+1)*s] = out
        
    return outs


def batch_grad(image_recon, coords, c, model):
    s = 64
    
    out_grads = np.zeros([np.shape(coords)[0], np.shape(coords)[1], 2])
    for i in range(np.shape(coords)[1]//s):
        
        batch_coords = coords[:,i*s: (i+1)*s]
        out = model(batch_coords, image_recon)
        out_grad = gradient(out, batch_coords).detach().cpu().numpy()
        out_grads[:,i*s: (i+1)*s] = out_grad
        
    return out_grads


def batch_laplace(image_recon, coords, c, model):
    s = 64
    
    out_laplaces = np.zeros([np.shape(coords)[0], np.shape(coords)[1],1])
    for i in range(np.shape(coords)[1]//s):
        
        batch_coords = coords[:,i*s: (i+1)*s]
        out = model(batch_coords, image_recon)
        out_laplace = laplace(out, batch_coords).detach().cpu().numpy()
        out_laplaces[:,i*s: (i+1)*s] = out_laplace
        
    return out_laplaces


def batch_grad_pde(image_recon, coords, c, model):
    s = 64
    
    out_grads = np.zeros([np.shape(coords)[0], np.shape(coords)[1], 2])
    for i in range(np.shape(coords)[1]//s):
        
        batch_coords = coords[:,i*s: (i+1)*s]
        out = model(batch_coords, image_recon)
        out_grad = gradient(out, batch_coords).detach().cpu().numpy()
        out_grads[:,i*s: (i+1)*s] = out_grad
        
    return out_grads




    
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    
def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen,:sidelen], axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen    
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords

def get_mgrid_unbalanced(sidelen1,sidelen2):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen1,:sidelen2], axis=-1)[None,...].astype(np.float32)
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    pixel_coords = pixel_coords/(pixel_coords.max(dim = 0)[0]+1)  
    pixel_coords -= 0.5
    return pixel_coords


def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.view(batch_size, channels, sidelen, sidelen)

    
def plot_sample_image(img_batch, ax):
    # plot the first item in batch
    img = lin2img(img_batch)[0].detach().cpu().numpy()
    img += 1
    img /= 2.
    img = np.clip(img, 0., 1.)
    ax.set_axis_off()
    ax.imshow(img)
    

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def multiple_gradient(y , x):
    out = torch.zeros([y.shape[0] , y.shape[1], 6])
    for i in range(y.shape[-1]):
        a = torch.autograd.grad(y[...,i], x, torch.ones_like(y[...,i]), create_graph=True)[0]
        out[...,i:i +2] = a
    return out

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def aeder_loss(x_true, x_hat, loss_type = 'mse', pyramid_l = None, mse_l = None, vgg = None):
    batch_size = x_true.shape[0]
    if loss_type == 'mse':
        loss = mse_l(x_hat.reshape(batch_size, -1) , x_true.reshape(batch_size, -1))

    elif loss_type == 'pyramid':
        loss = pyramid_l(x_hat , x_true)

    elif loss_type == 'style':

        reg_weight = 1e-6
        style_weight = 1
        feature_weight = 10
        pure_feat = vgg(x_true)
        recon_feat = vgg(x_hat)

        loss_style = 0
        loss_feature = 0

        for k in range(len(pure_feat)):

            bs, ch, h, w = pure_feat[k].size()

            pure_re_feat= pure_feat[k].view(bs, ch, h*w)
            gram_pure = torch.matmul(pure_re_feat, torch.transpose(pure_re_feat,1,2))/(ch*h*w)

            recon_re_feat= recon_feat[k].view(bs, ch, h*w)
            gram_recon = torch.matmul(recon_re_feat, torch.transpose(recon_re_feat,1,2))/(ch*h*w)

            loss_style = loss_style+ mse_l(gram_pure.view(batch_size,-1),gram_recon.view(batch_size,-1))

            loss_feature = loss_feature + mse_l(pure_feat[k].reshape(batch_size,-1),recon_feat[k].reshape(batch_size,-1))#/(pure_feat[k].size(1)*pure_feat[k].size(2)*pure_feat[k].size(3))

        
            loss_style = style_weight * loss_style
            loss_feature = feature_weight * loss_feature

            loss_tv = reg_weight * (
                torch.sum(torch.abs(x_hat[:, :, :, :-1] - x_hat[:, :, :, 1:])) +
                torch.sum(torch.abs(x_hat[:, :, :-1, :] - x_hat[:, :, 1:, :])))

            loss = loss_feature + loss_style + loss_tv

    elif loss_type == 'style_mse':

        reg_weight = 1e-6
        style_weight = 1
        feature_weight = 10
        mse_weight = 5000
        pure_feat = vgg(x_true)
        recon_feat = vgg(x_hat)

        loss_style = 0
        loss_feature = 0

        for k in range(len(pure_feat)):

            bs, ch, h, w = pure_feat[k].size()

            pure_re_feat= pure_feat[k].view(bs, ch, h*w)
            gram_pure = torch.matmul(pure_re_feat, torch.transpose(pure_re_feat,1,2))/(ch*h*w)

            recon_re_feat= recon_feat[k].view(bs, ch, h*w)
            gram_recon = torch.matmul(recon_re_feat, torch.transpose(recon_re_feat,1,2))/(ch*h*w)

            loss_style = loss_style+ mse_l(gram_pure.view(batch_size,-1),gram_recon.view(batch_size,-1))

            loss_feature = loss_feature + mse_l(pure_feat[k].reshape(batch_size,-1),recon_feat[k].reshape(batch_size,-1))#/(pure_feat[k].size(1)*pure_feat[k].size(2)*pure_feat[k].size(3))

        
            loss_style = style_weight * loss_style
            loss_feature = feature_weight * loss_feature

            loss_tv = reg_weight * (
                torch.sum(torch.abs(x_hat[:, :, :, :-1] - x_hat[:, :, :, 1:])) +
                torch.sum(torch.abs(x_hat[:, :, :-1, :] - x_hat[:, :, 1:, :])))

            loss_mse = mse_weight * mse_l(x_hat.reshape(batch_size, -1) , x_true.reshape(batch_size, -1))

            loss = loss_feature + loss_style + loss_tv + loss_mse


    return loss





def training_strategy(x_true, image_size, factor = None , mode = 'continuous', image_recon = None):

    image_recon = x_true if image_recon == None else image_recon

    if mode == 'continuous':

        image_size_random = np.random.randint(low = image_size//4, high = image_size//2, size = 1)[0]
        x_low = F.interpolate(image_recon, size = image_size_random, antialias = True, mode = 'bilinear')
        x_high = x_true
        image_size_high = image_size


    elif mode == 'factor':

        image_size_low = np.random.randint(low = image_size//8, high = image_size//2, size = 1)[0]
        image_size_high = 2 * image_size_low
        x_high = F.interpolate(x_true, size = image_size_high, antialias = True, mode = 'bilinear')
        x_low = F.interpolate(image_recon, size = image_size_low, antialias = True, mode = 'bilinear')

    elif mode == 'single':
        x_high = x_true
        x_low = F.interpolate(image_recon, size = image_size//2, antialias = True, mode = 'bilinear')
        image_size_high = image_size

    return x_high, x_low, image_size_high


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained = True).features)[:23]
        self.features = torch.nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results.append(x)

        return results



def fbp_batch(x):
    n_measure = x.shape[2]
    theta = np.linspace(0., 180., n_measure, endpoint=False)

    fbps = []
    for i in range(x.shape[0]):
        fbps.append(iradon(x[i], theta=theta, circle = False))

    fbps = np.array(fbps)
    return fbps