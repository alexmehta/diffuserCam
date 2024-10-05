import torch
import torch.fft as fft
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml
from op import ImageReconstructor, loaddata
from tqdm import trange
import imageio
torch.manual_seed(0)
def PnPReconstruction(meas_func, x, iters=1000):
    import deepinv
    from deepinv.utils.plotting import plot, plot_curves
    class Phys(deepinv.physics.LinearPhysics):
        def __init__(self, process):
            super().__init__()
            self.process = process
        def forward(self, x):
            return self.process(x)
    phys = Phys(meas_func)
    y = phys(x)
    prior = deepinv.optim.prior.PnP(denoiser=deepinv.models.DnCNN(in_channels=1, out_channels=1, pretrained="download").to(y.device))
    model = deepinv.optim.optimizers.optim_builder(iteration='PGD', prior=prior, data_fidelity=deepinv.optim.data_fidelity.L2(), max_iter=iters, verbose=True).to(y.device).eval()
    
    result, metrics = model(y.unsqueeze(dim=0), phys, x_gt=x.unsqueeze(dim=0), compute_metrics=True)
    # mse between the final result and the ground truth
    print(f"Final MSE: {torch.nn.functional.mse_loss(result, x).item()}")
    plot_curves(metrics, save_dir="curves/", show=False)

    return result.squeeze(dim=0).squeeze(dim=0), result
    
def grad_descent(h, b, iters, save_interval):
    model = ImageReconstructor(h).to(h.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    def non_neg(xi):
        return torch.clamp(xi, min=0)
    
    frames = []
    for iteration in trange(iters):
        optimizer.zero_grad()
        output = model()
        loss = torch.nn.functional.mse_loss(output, b)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.v.data = non_neg(model.v.data)
        if iteration % save_interval == 0 or iteration == iters - 1:
            print(f"Iteration {iteration}, Loss: {loss.item()}")
            image = model.crop()
            frame = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
    
    return model.crop().detach(), frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Reconstruct an image from a PSF')
    parser.add_argument('--psf_path', type=str, default='./images/psf_sample.tif', help='Path to the PSF image')
    parser.add_argument('--img_path', type=str, default='./images/rawdata_hand_sample.tif', help='Path to the image to reconstruct')
    parser.add_argument('--f', type=float, default=1/2**4, help='Factor to resize the images by')
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations to run the optimization for')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving frames')
    parser.add_argument('--gif_save', type=bool, default=True, help='Save a gif', required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mps', type=bool, default=False, help='Use MPS', required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--show_imgs', type=bool, default=False, help='Show images', required=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    device = torch.device("mps" if (torch.backends.mps.is_available() and args.mps) else "cpu")
    psf, data = loaddata(args.psf_path, args.img_path, args.f, show_im=False)
    psf, data = psf.to(device), data.to(device)
    original_img = data
    # if args.show_imgs:
    #     plt.imshow(data.cpu().numpy(), cmap='gray')
    #     plt.title('Original GT')
    #     plt.show()
    # data = ImageReconstructor(psf).to(device).simulate_measurement(data, noise_level=0.0000000).to(device)
    # data /= torch.norm(data.flatten())
    # if args.show_imgs:
    #     plt.imshow(data.cpu().numpy(), cmap='gray')
    #     plt.title('Raw data')
    #     plt.show()
    final_im, frames = PnPReconstruction(lambda x: ImageReconstructor(psf).to(device).simulate_measurement(x, noise_level=0.000000001).to(device), original_img, args.iters)

    # final_im, frames = grad_descent(psf, data, args.iters, args.save_interval)
    # if args.gif_save:
        # imageio.mimsave('reconstruction.gif', frames, format='GIF', duration=0.1) 
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img.cpu().numpy(), cmap='gray')
    plt.title('Original GT')
    plt.subplot(1, 2, 2)
    plt.imshow(final_im.detach().cpu().numpy(), cmap='gray')
    plt.title(f'Final reconstruction after {args.iters} iterations')
    plt.savefig('comparison.png', bbox_inches='tight')
    if args.show_imgs:
        plt.show()