import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import ImageFolder
from utils import *
import config_funknn as config


class Dataset_loader(torch.utils.data.Dataset):
    def __init__(self, dataset , size=(1024,1024), c = 3, quantize = False):

        if c==1:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
            ])

        self.c = c
        self.dataset = dataset
        self.meshgrid = get_mgrid(size[0])
        self.im_size = size
        self.quantize = quantize

        if self.dataset == 'train':
            self.img_dataset = ImageFolder(config.train_path, self.transform)

        elif self.dataset == 'test':
            self.img_dataset = ImageFolder(config.test_path, self.transform)

        elif self.dataset == 'ood':
            lsun_class = ['bedroom_val']
            self.img_dataset = torchvision.datasets.LSUN(config.ood_path,
                classes=lsun_class, transform=self.transform)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img = self.img_dataset[item][0]
        img = transforms.ToPILImage()(img)
        img = self.transform(img).permute([1,2,0])
        img = img.reshape(-1, self.c)

        if self.quantize:
            img = img * 255.0
            img = torch.multiply(8.0, torch.div(img , 8 , rounding_mode = 'floor'))
            img = img/255.0
        
        return img
    
