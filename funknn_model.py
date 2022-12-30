import torch.nn as nn
import torch.nn.functional as F
import torch


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

def grid_sample_customized(image, grid, pad = 'reflect', align_corners = False):
    '''Differentiable grid_sample:
    equivalent performance with torch.nn.functional.grid_sample can be obtained by setting
    align_corners = True,
    pad: 'border': use border pixels,
    'reflect': create reflect pad manually.

    image is a tensor of shape (N, C, IH, IW)
    grid is a tensor of shape (N, H, W, 2)'''

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



class FunkNN(nn.Module):
    '''FunkNN module'''

    def __init__(self, c):
        super(FunkNN, self).__init__()
        
        self.c = c

        CNNs = []
        prev_ch = self.c
        num_layers = [64,64,64,64,64,64,64,64]
        for i in range(len(num_layers)):
            CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,2,
                padding = 'same', bias = True))
            prev_ch = num_layers[i]

        self.CNNs = nn.ModuleList(CNNs)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(2 * 2 * 64, 64, bias = True)
        self.linear2 = nn.Linear(64, 64, bias = True)
        self.linear3 = nn.Linear(64, 64, bias = True)
        self.linear4 = nn.Linear(64, self.c, bias = True)

        # Adaptive receptive field
        ws1 = torch.ones(1)
        self.ws1 = nn.Parameter(ws1.clone().detach(), requires_grad=True)
        ws2 = torch.ones(1)
        self.ws2 = nn.Parameter(ws2.clone().detach(), requires_grad=True)

        alpha1 = torch.zeros(1)
        self.alpha1 = nn.Parameter(alpha1.clone().detach(), requires_grad=False)
        alpha2 = torch.zeros(1)
        self.alpha2 = nn.Parameter(alpha2.clone().detach(), requires_grad=False)


    def cropper(self, image, coordinate , output_size):
        '''Cropper using Spatial Transformer'''
        # Coordinate shape: b X b_pixels X 2
        # image shape: b X b_pixels X c X h X w
        d_coordinate = coordinate * 2
        b , b_pixels , c , h , w = image.shape
        crop_size = 2 * output_size/h
        x_m_x = crop_size/2
        x_p_x = d_coordinate[:,:,1]
        y_m_y = crop_size/2
        y_p_y = d_coordinate[:,:,0]
        theta = torch.zeros(b, b_pixels, 2,3).to(image.device)
        theta[:,:,0,0] = x_m_x * self.ws1
        theta[:,:,0,2] = x_p_x
        theta[:,:,1,1] = y_m_y * self.ws2
        theta[:,:,1,2] = y_p_y
        theta[:,:,0,1] = self.alpha1
        theta[:,:,1,0] = self.alpha2

        image = image.reshape(b*b_pixels , c , h , w)
        theta = theta.reshape(b*b_pixels , 2 , 3)

        f = F.affine_grid(theta, size=(b * b_pixels, c, output_size, output_size), align_corners=False)
        # Non-differentiable grid_sampler
        # image_cropped = F.grid_sample(image, f, mode = 'bicubic', align_corners=False, padding_mode='reflection')
        # Differentiable grid_sampler, well-suited for solving PDEs
        image_cropped = grid_sample_customized(image, f, pad = 'reflect', align_corners = False)

        return image_cropped

       
    def forward(self, coordinate, x):
        b , b_pixels , _ = coordinate.shape
        x = torch.unsqueeze(x, dim = 1)
        x =x.expand(-1, b_pixels , -1, -1, -1)

        w_size = 9 # patch size
        x = self.cropper(x , coordinate , output_size = w_size)
        mid_pix = x[:,:,4,4] # Centeric pixel

        for i in range(len(self.CNNs)):

            x_temp = x
            x = self.CNNs[i](x)
            x = F.relu(x)
            if i % 4 == 3:
                x = self.maxpool(x)
            else:
                if not (i ==0 or i == len(self.CNNs)-1):
                    x = x+x_temp # Internal skip connection

        x = torch.flatten(x, 1)

        x_tmp = F.relu(self.linear1(x)) #+ mid_pix
        x_tmp = F.relu(self.linear2(x_tmp)) + x_tmp
        x_tmp = F.relu(self.linear3(x_tmp)) + x_tmp
        x = self.linear4(x_tmp) + mid_pix # external skip connection to the centric pixel
        x = x.reshape(b, b_pixels, -1)

        return x




