import torch.nn as nn
import torch.nn.functional as F
import torch
import config_funknn as config
import numpy as np


def squeeze(x , f):
    x = x.permute(0,2,3,1)
    b, N1, N2, nch = x.shape
    x = torch.reshape(
        torch.permute(
            torch.reshape(x, shape=[b, N1//f, f, N2//f, f, nch]),
            [0, 1, 3, 2, 4, 5]),
        [b, N1//f, N2//f, nch*f*f])
    x = x.permute(0,3,1,2)
    return x


def reflect_coords(ix, min_val, max_val):

    pos_delta = ix[ix>max_val] - max_val

    neg_delta = min_val - ix[ix < min_val]

    ix[ix>max_val] = ix[ix>max_val] - 2*pos_delta
    ix[ix<min_val] = ix[ix<min_val] + 2*neg_delta

    return ix


def cubic_coords(ix,iy, indices =  [-1,0,1,2]):

    with torch.no_grad():
        ix_base = torch.floor(ix)
        iy_base = torch.floor(iy)
        points = torch.zeros(ix.shape + (len(indices)**2,2)).to(ix.device)
        for i in range(len(indices)):
            for j in range(len(indices)):
                points[...,len(indices) *i + j,0] = indices[i] + ix_base
                points[...,len(indices) *i + j,1] = indices[j] + iy_base
    
    return points


def cubic_kernel(s1, order = 4):
    s = s1 + 1e-6
    out = torch.zeros_like(s)

    if order == 4:

        p = torch.abs(s[torch.abs(s)< 2])
        out[torch.abs(s)< 2] = -0.5 * p**3 + 2.5 * p**2 -4*p + 2

        p = torch.abs(s[torch.abs(s)== 1])
        out[torch.abs(s)== 1] = ((1.5 * p**3 - 2.5 * p**2 + 1) + 3*(-0.5 * p**3 + 2.5 * p**2 -4*p + 2))/4

        p = torch.abs(s[torch.abs(s)< 1])
        p_wo_abs = s[torch.abs(s)< 1]
        out[torch.abs(s)< 1] = 1.5 * p**3 - 2.5 * p_wo_abs**2 + 1

    elif order == 6:
        p = torch.abs(s[torch.abs(s)< 3])
        out[torch.abs(s)< 3] = (1 * p**3)/12 - (2 * p**2)/3 + 21*p/12 - 1.5
        p = torch.abs(s[torch.abs(s)< 2])
        out[torch.abs(s)< 2] = -(7 * p**3)/12 + 3 * p**2 -59*p/12 + 15/6
        p = torch.abs(s[torch.abs(s)< 1])
        out[torch.abs(s)< 1] = (4 * p**3)/3 - (7 * p**2)/3 + 1

    return out



class FunkNN(nn.Module):
    '''FunkNN module'''

    def __init__(self, c):
        super(FunkNN, self).__init__()
        
        self.c = c
        self.w_size = 9 # patch size

        if config.network == 'CNN':
            CNNs = []
            prev_ch = self.c
            num_layers = [64,64,64,64,64,64,64,64]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,2,
                    padding = 'same', bias = True))
                prev_ch = num_layers[i]

            self.CNNs = nn.ModuleList(CNNs)
            self.maxpool = nn.MaxPool2d(2, 2)
            prev_unit = 2 * 2 * 64
            hidden_units = [64,64,64,self.c]
        
        if config.network == 'MLP':
            prev_unit = self.w_size * self.w_size * self.c
            hidden_units = [512,512,256,256,128,128,64,64,64,self.c]

        fcs = []
        for i in range(len(hidden_units)):
            fcs.append(nn.Linear(prev_unit, hidden_units[i], bias = True))
            prev_unit = hidden_units[i]

        self.MLP = nn.ModuleList(fcs)

        if config.activation == 'sin':
            w0 = 30.0
            w = []
            for i in range(len(self.MLP)):
                w_init = torch.ones(1) * w0
                w.append(nn.Parameter(w_init.clone().detach(), requires_grad=True))
                w_shape = self.MLP[i].weight.data.shape
                b_shape = self.MLP[i].bias.data.shape
                w_std = (1 / w_shape[1]) if i==0 else (np.sqrt(6.0 / w_shape[1]) / w0)
                # w_std = (1 / w_shape[1])
                self.MLP[i].weight.data = (2 * torch.rand(w_shape) - 1) * w_std
                self.MLP[i].bias.data = (2 * torch.rand(b_shape) - 1) * w_std
            self.w = nn.ParameterList(w)

        # Adaptive receptive field
        ws1 = torch.ones(1)
        self.ws1 = nn.Parameter(ws1.clone().detach(), requires_grad=True)
        ws2 = torch.ones(1)
        self.ws2 = nn.Parameter(ws2.clone().detach(), requires_grad=True)



    def grid_sample_customized(self, image, grid, mode = 'cubic_conv' , pad = 'reflect', align_corners = False):
        '''Differentiable grid_sample:
        equivalent performance with torch.nn.functional.grid_sample can be obtained by setting
        align_corners = True,
        pad: 'border': use border pixels,
        'reflect': create reflect pad manually.
        image is a tensor of shape (N, C, IH, IW)
        grid is a tensor of shape (N, H, W, 2)'''

        if mode == 'bilinear':
            N, C, IH, IW = image.shape
            _, H, W, _ = grid.shape

            ix = grid[..., 0]
            iy = grid[..., 1]


            if align_corners == True:
                ix = ((ix + 1) / 2) * (IW-1);
                iy = ((iy + 1) / 2) * (IH-1);
                
                boundary_x = (0, IW-1)
                boundary_y = (0, IH-1)
                
            
            elif align_corners == False:
                ix = ((1+ix)*IW/2) - 1/2
                iy = ((1+iy)*IH/2) - 1/2
                
                boundary_x = (-1/2, IW-1/2)
                boundary_y = (-1/2, IH-1/2)
            

            with torch.no_grad():
                ix_nw = torch.floor(ix);
                iy_nw = torch.floor(iy);
                ix_ne = ix_nw + 1;
                iy_ne = iy_nw;
                ix_sw = ix_nw;
                iy_sw = iy_nw + 1;
                ix_se = ix_nw + 1;
                iy_se = iy_nw + 1;

            nw = (ix_se - ix)    * (iy_se - iy)
            ne = (ix    - ix_sw) * (iy_sw - iy)
            sw = (ix_ne - ix)    * (iy    - iy_ne)
            se = (ix    - ix_nw) * (iy    - iy_nw)

            if pad == 'reflect' or 'reflection':
                
                ix_nw = reflect_coords(ix_nw, boundary_x[0], boundary_x[1])
                iy_nw = reflect_coords(iy_nw, boundary_y[0], boundary_y[1])

                ix_ne = reflect_coords(ix_ne, boundary_x[0], boundary_x[1])
                iy_ne = reflect_coords(iy_ne, boundary_y[0], boundary_y[1])

                ix_sw = reflect_coords(ix_sw, boundary_x[0], boundary_x[1])
                iy_sw = reflect_coords(iy_sw, boundary_y[0], boundary_y[1])

                ix_se = reflect_coords(ix_se, boundary_x[0], boundary_x[1])
                iy_se = reflect_coords(iy_se, boundary_y[0], boundary_y[1])


            elif pad == 'border':

                with torch.no_grad():
                    torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
                    torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

                    torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
                    torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

                    torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
                    torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

                    torch.clamp(ix_se, 0, IW-1, out=ix_se)
                    torch.clamp(iy_se, 0, IH-1, out=iy_se)


            image = image.reshape(N, C, IH * IW)

            nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
            ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
            sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
            se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

            out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
                    ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
                    sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
                    se_val.view(N, C, H, W) * se.view(N, 1, H, W))


            return out_val


        elif mode == 'cubic_conv':

            N, C, IH, IW = image.shape
            _, H, W, _ = grid.shape

            ix = grid[..., 0]
            iy = grid[..., 1]


            if align_corners == True:
                ix = ((ix + 1) / 2) * (IW-1);
                iy = ((iy + 1) / 2) * (IH-1);
                
                boundary_x = (0, IW-1)
                boundary_y = (0, IH-1)
                
            
            elif align_corners == False:
                ix = ((1+ix)*IW/2) - 1/2
                iy = ((1+iy)*IH/2) - 1/2
                
                boundary_x = (-1/2, IW-1/2)
                boundary_y = (-1/2, IH-1/2)
            
            # indices = [-2,-1,0,1,2,3] # order 6, slower but more precise
            indices = [-1,0,1,2] # order 4

            points = cubic_coords(ix,iy, indices)
            n_neighrbours = len(indices)**2

            ix_relative = ix.unsqueeze(dim = -1)- points[...,0]
            iy_relative = iy.unsqueeze(dim = -1)- points[...,1]

            points[...,0] = reflect_coords(points[...,0], boundary_x[0], boundary_x[1])
            points[...,1] = reflect_coords(points[...,1], boundary_y[0], boundary_y[1])
            points = points.unsqueeze(dim = 1).expand(-1,C,-1,-1,-1,-1)

            image = image.reshape(N, C, IH * IW)
            points_values = torch.gather(image,2, (points[...,1] * IW + points[...,0]).long().view(N,C,H*W*n_neighrbours))
            points_values = points_values.reshape(N,C,H,W,n_neighrbours)

            ux = cubic_kernel(ix_relative, order = len(indices)).unsqueeze(dim = 1)
            uy = cubic_kernel(iy_relative, order = len(indices)).unsqueeze(dim = 1)

            recons = points_values * ux * uy
            out_val = torch.sum(recons, dim = 4)

            return out_val


    def cropper(self, image, coordinate , output_size):
        '''Cropper using Spatial Transformer'''
        # Coordinate shape: b X b_pixels X 2
        # image shape: b X b_pixels X c X h X w
        d_coordinate = coordinate * 2
        b , c , h , w = image.shape
        b_pixels = coordinate.shape[1]
        crop_size = 2 * (output_size-1)/(h-1)
        x_m_x = crop_size/2
        x_p_x = d_coordinate[:,:,1]
        y_m_y = crop_size/2
        y_p_y = d_coordinate[:,:,0]
        theta = torch.zeros(b, b_pixels, 2,3).to(image.device)
        theta[:,:,0,0] = x_m_x * self.ws1
        theta[:,:,0,2] = x_p_x
        theta[:,:,1,1] = y_m_y * self.ws2
        theta[:,:,1,2] = y_p_y

        theta = theta.reshape(b*b_pixels , 2 , 3)

        f = F.affine_grid(theta, size=(b * b_pixels, c, output_size, output_size), align_corners=False)
        f = f.reshape(b, b_pixels , output_size, output_size,2)
        f = f.reshape(b, b_pixels * output_size, output_size,2)

        if config.interpolation_kernel == 'bicubic':
            # Non-differentiable grid_sampler
            image_cropped = F.grid_sample(image, f, mode = 'bicubic', align_corners=False, padding_mode='reflection')
        
        else:
            # Differentiable grid_sampler, well-suited for solving PDEs
            # mode = 'bilinear' is fast but can only be used for PDEs with the fiesr-order derivatives.
            # mode = 'cubic_conv' is slow but can be used for solving PDEs with first- and second-order derivatives.
            image_cropped = self.grid_sample_customized(image, f, mode = config.interpolation_kernel)

        image_cropped = image_cropped.permute(0,2,3,1)
        image_cropped = image_cropped.reshape(b, b_pixels , output_size, output_size,c)
        image_cropped = image_cropped.reshape(b* b_pixels , output_size, output_size,c)
        image_cropped = image_cropped.permute(0,3,1,2)

        return image_cropped

       
    def forward(self, coordinate, x):
        b , b_pixels , _ = coordinate.shape

        x = self.cropper(x , coordinate , output_size = self.w_size)
        mid_pix = x[:,:,4,4] # Centeric pixel

        if config.network == 'CNN':
            for i in range(len(self.CNNs)):

                x_temp = x
                x = self.CNNs[i](x)
                
                if config.activation == 'sin': 
                    x = torch.sin(x) # Sin activations for more accurate derivatives
                else:
                    x = F.relu(x)

                if i % 4 == 3:
                    x = self.maxpool(x)
                else:
                    if not (i ==0 or i == len(self.CNNs)-1):
                        x = x+x_temp # Internal skip connection

        x = torch.flatten(x, 1)

        for i in range(len(self.MLP)-1):
            if config.activation == 'sin':
                x = torch.sin(self.w[i] * self.MLP[i](x)) # Sin activations for more accurate derivatives
            else:
                x = F.relu(self.MLP[i](x))
                
        x = self.MLP[-1](x) + mid_pix # external skip connection to the centric pixel
        x = x.reshape(b, b_pixels, -1)

        return x




