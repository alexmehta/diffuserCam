import numpy as np
# import numpy.fft as fft
import torch.fft as fft
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
import torch
from torch import nn
import torch
import torch.fft as fft
from torch import nn
import torch.nn.functional as F

def loaddata(psfname, imgname, f, show_im=True):
    psf = Image.open(psfname)
    psf = torch.tensor(np.array(psf), dtype=torch.float32)
    data = Image.open(imgname)
    data = torch.tensor(np.array(data), dtype=torch.float32)
    
    # if image is RGB, convert to grayscale
    if len(data.shape) == 3:
        data = torch.mean(data, dim=2)
    if data.shape != psf.shape:
        data = F.interpolate(data.unsqueeze(0).unsqueeze(0), 
                             size=psf.shape, 
                             mode='bilinear', 
                             align_corners=False).squeeze(0).squeeze(0)
    
    bg = torch.mean(psf[5:15,5:15])
    psf -= bg
    data -= bg
    def resize(img, factor):
        num = int(-torch.log2(torch.tensor(factor)).item())
        for _ in range(num):
            img = 0.25 * (img[::2,::2,...] + img[1::2,::2,...] + img[::2,1::2,...] + img[1::2,1::2,...])
        return img
    
    psf = resize(psf, f)
    data = resize(data, f)
    
    psf /= torch.norm(psf.flatten())
    data /= torch.norm(data.flatten())
    
    # if show_im:
    #     fig1 = plt.figure()
    #     plt.imshow(psf.cpu().numpy(), cmap='gray')
    #     plt.title('PSF')
    #     plt.show()
    #     fig2 = plt.figure()
    #     plt.imshow(data.cpu().numpy(), cmap='gray')
    #     plt.title('Raw data')
    #     plt.show()
    return psf, data

def nextPow2(n):
    return int(2**torch.ceil(torch.log2(torch.tensor(n, dtype=torch.float))))

class ImageReconstructor(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        # this is just initMatricies from tutorial notebook
        self.h = h
        self.init_shape = h.shape
        self.padded_shape = [nextPow2(2*n - 1) for n in self.init_shape]
        self.starti = (self.padded_shape[0] - self.init_shape[0]) // 2
        self.endi = self.starti + self.init_shape[0]
        self.startj = (self.padded_shape[1] // 2) - (self.init_shape[1] // 2)
        self.endj = self.startj + self.init_shape[1]
        hpad = torch.zeros(self.padded_shape, device=h.device)
        hpad[self.starti:self.endi, self.startj:self.endj] = h
        self.H = fft.fft2(hpad, norm="ortho")
        self.v = torch.nn.Parameter(self.H.real.detach().clone(), requires_grad=True)
    def forward(self):
        Vk = fft.fft2(self.v, norm="ortho")
        reconstruct = fft.ifftshift(fft.ifft2(self.H*Vk, norm="ortho"))
        return reconstruct[self.starti:self.endi, self.startj:self.endj].real
    def crop(self):
        return self.v[self.starti:self.endi, self.startj:self.endj].real
    def simulate_measurement(self, input_image, noise_level=0.01):
        if input_image.shape != self.init_shape:
            print("in future please resize the image before passing it to simulate_measurement")
            input_image = F.interpolate(input_image.unsqueeze(0).unsqueeze(0), 
                                    size=self.init_shape, 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(0).squeeze(0)

        padded_input = torch.zeros(self.padded_shape, device=input_image.device)
        padded_input[self.starti:self.endi, self.startj:self.endj] = input_image

        input_fft = fft.fft2(padded_input, norm="ortho")
        blurred_fft = self.H * input_fft
        blurred = fft.ifftshift(fft.ifft2(blurred_fft, norm="ortho")).real
        blurred = blurred[self.starti:self.endi, self.startj:self.endj]
        noise = torch.randn_like(blurred) * noise_level
        measured = blurred + noise
        return measured