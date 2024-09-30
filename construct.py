import torch
import torch.fft as fft
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml
from op import ImageReconstructor, loaddata
from tqdm import trange
import imageio

def grad_descent(h, b, iters, save_interval):
    model = ImageReconstructor(h).to(h.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    def non_neg(xi):
        return torch.clamp(xi, min=0)
    
    frames = []
    for iteration in trange(iters):
        if iteration % save_interval == 0:
            # print(f"Iteration {iteration}, Loss: {loss.item()}")
            image = model.crop()
            frame = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        optimizer.zero_grad()
        output = model()
        loss = torch.nn.functional.mse_loss(output, b)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.v.data = non_neg(model.v.data)
       
    
    return model.crop().detach(), frames

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct an image from a PSF')
    parser.add_argument('--psf_path', type=str, default='./images/psf_sample.tif', help='Path to the PSF image')
    parser.add_argument('--img_path', type=str, default='./images/rawdata_hand_sample.tif', help='Path to the image to reconstruct')
    parser.add_argument('--f', type=float, default=1/8, help='Factor to resize the images by')
    parser.add_argument('--iters', type=int, default=1000, help='Number of iterations to run the optimization for')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving frames')
    parser.add_argument('--gif_save', type=bool, default=True, help='Save a gif', required=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    psf, data = loaddata(args.psf_path, args.img_path, args.f, show_im=True)
    psf, data = psf.to(device), data.to(device)

    final_im, frames = grad_descent(psf, data, args.iters, args.save_interval)

    # Save as GIF
    if args.gif_save:
        imageio.mimsave('reconstruction.gif', frames, duration=1) 
    print(f"Reconstruction process saved as 'reconstruction.gif'")

    # Save final image
    plt.figure(figsize=(10, 10))
    plt.imshow(final_im.cpu().numpy(), cmap='gray')
    plt.title(f'Final reconstruction after {args.iters} iterations')
    plt.axis('off')
    plt.savefig('final_reconstruction.png', bbox_inches='tight')
    print("Final reconstruction saved as 'final_reconstruction.png'")