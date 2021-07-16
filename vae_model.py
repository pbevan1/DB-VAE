"""
Here the structure of the network is made in pytorch
"""

from typing import List, Union, Optional
import torch
import os
from logger import logger
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.stats import norm
from setup import *

class Encoder(nn.Module):
    """
    Encodes the data using a CNN
    Input => 64x64 image
    Output => mean vector z_dim
              log_std vector z_dim
              predicted value
    """

    def __init__(self, z_dim: int = 20, custom_layers: Optional[nn.Sequential] = None):
        super().__init__()

        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 2048, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2048),

            nn.Flatten(),

            nn.Linear(2048, 1000),
            nn.LeakyReLU(),

            nn.Linear(1000, z_dim*2+1)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, input: torch.Tensor):
        """
        Perform forward pass of encoder.
        """
        # print(f'Encoder input shape: {input.shape}')
        out = self.layers(input)
        # print(f'Encoder out shape: {out.shape}')

        sigout = self.sigmoid(out)

        # return classification, mean and log_std
        return out[:, 0], out[:, 1:self.z_dim+1], F.softplus(out[:,self.z_dim+1:]), sigout ################## adjust rest of code to use this


class UnFlatten(nn.Module):
    def __init__(self, channel_size, image_size):
        super(UnFlatten, self).__init__()
        self.channel_size = channel_size
        self.image_size = image_size

    def forward(self, input):
        # print(f'out shape: {input.shape}')
        return input.view(-1, self.channel_size, self.image_size, self.image_size)

class Decoder(nn.Module):
    """
    Encodes the data using a CNN
    Input => sample vector z_dim
    Output => 64x64 image
    4 6 13 29 61
    """

    def __init__(self, z_dim: int = 20, custom_layers: Optional[nn.Sequential] = None):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 2048*1*1),
            UnFlatten(2048, 1),

            nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),

            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor):
        """
        Perform forward pass of encoder.
        """

        # print(f'Decoder in shape: {input.shape}')
        out = self.layers(input)
        # print(f'Decoder out shape: {out.shape}')

        return out


class Db_vae(nn.Module):
    def __init__(
        self,
        config=None,
        z_dim: int = 20,
        hist_size: int = 1000,
        alpha: float = 0.01,
        num_bins: int = 10,
        device: str = "cpu",
        custom_encoding_layers: Optional[nn.Sequential] = None,
        custom_decoding_layers: Optional[nn.Sequential] = None
    ):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.config = config
        self.run_mode = self.config.run_mode
        self.encoder = Encoder(z_dim, custom_encoding_layers)
        self.decoder = Decoder(z_dim, custom_decoding_layers)

        if ARGS.DP:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.target_dist = torch.distributions.normal.Normal(0, 1)

        self.c1 = 1
        self.c2 = 1
        self.c3 = 0.1

        self.num_bins = num_bins
        self.min_val = -15
        self.max_val = 15
        self.xlin = np.linspace(self.min_val, self.max_val, self.num_bins).reshape(1,1,self.num_bins)
        self.hist = np.zeros((z_dim, self.num_bins))
        self.means = torch.Tensor().to(self.device)
        self.std = torch.Tensor().to(self.device)

        self.alpha = alpha

    @staticmethod
    def init(config, path_to_model: str, device: str, z_dim: int):
        full_path_to_model = f"results/weights/{path_to_model}/model.pt"
        if not os.path.exists(full_path_to_model):
            logger.error(
                f"Can't find model at {full_path_to_model}",
                next_step="Evaluation will stop",
                tip="Double check your path to model"
            )
            raise Exception

        model: Db_vae = Db_vae(config=config, z_dim=z_dim, device=device)

        try:
            model.load_state_dict(torch.load(full_path_to_model, map_location=device))
        except:
            logger.error("Unable to load model from {full_path_to_model}.",
                        next_step="Model will not initialize",
                        tip="Did you use the right config parameters, or custom layers from the stored model?"
            )

        logger.info(f"Loaded model from {path_to_model}!")
        return model


    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Given images, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        pred, mean, std, sigout = self.encoder(images)

        loss_class = F.binary_cross_entropy_with_logits(pred, labels.float(), reduction='none')

        # We only want to calculate the loss towards actual faces
        # faceslicer = labels == 1
        # facemean = mean #[faceslicer]
        # facestd = std #[faceslicer]

        # Get single samples from the distributions with reparametrisation trick
        dist = torch.distributions.normal.Normal(mean, std)
        z = dist.rsample().to(self.device)

        # if self.run_mode == 'perturb':
        #     z = z[0]*2

        res = self.decoder(z)


        # calculate VAE losses
        loss_recon = (images - res)**2
        loss_recon = loss_recon.view(loss_recon.shape[0],-1).mean(1)

        loss_kl = torch.distributions.kl.kl_divergence(dist, self.target_dist)
        loss_kl = loss_kl.view(loss_kl.shape[0],-1).mean(1)

        loss_vae = self.c2 * loss_recon + self.c3 * loss_kl
        loss_total = self.c1 * loss_class

        # # Only add loss to positions of faces, rest is zero
        # zeros = torch.zeros(faceslicer.shape[0]).to(self.device)
        # zeros[faceslicer] = loss_vae

        loss_total = loss_total + loss_vae #+ zeros

        return pred, loss_total, sigout

    def forward_eval(self, images: torch.Tensor):
        """
        Given images, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        with torch.no_grad():
            pred, _,_, sigout = self.encoder(images)

        return pred, sigout


    def perturb_var(self, images: torch.Tensor, amount: int, var_to_perturb):
        with torch.no_grad():

            _, mean, std, _ = self.encoder(images)

            print(f'mean shape: {mean.shape}')
            print(f'std shape: {std.shape}')

            mean_1, std_1 = mean[0,var_to_perturb], std[0,var_to_perturb]
            mean_2, std_2 = mean[1,var_to_perturb], std[1,var_to_perturb]

            all_mean  = torch.tensor([]).to(self.device)
            all_std = torch.tensor([]).to(self.device)

            print(f'mean_1: {mean_1}')
            print(f'mean_2: {mean_2}')

            diff_mean = mean_1 - mean_2
            diff_std = std_1 - std_2

            steps_mean = diff_mean / (amount-1)
            steps_std = diff_std / (amount-1)

            print(f'steps_mean: {steps_mean}')
            print(f'steps_std: {steps_std}')

            for i in range(amount):
                mean[0, var_to_perturb] = mean[0, var_to_perturb] - steps_mean * i
                std[0, var_to_perturb] = std[0, var_to_perturb] - steps_std * i
                all_mean = torch.cat((all_mean, mean[0, :]))
                all_std = torch.cat((all_std, std[0, :]))
                # print(all_mean.shape)
                # print(all_std.shape)

            all_mean = all_mean.view(amount, -1)
            all_std = all_std.view(amount, -1)

            print(f'all_mean shape: {all_mean.shape}')
            print(f'all_std shape: {all_std.shape}')

            # print(all_mean)
            # print(all_std)

            dist = torch.distributions.normal.Normal(all_mean, all_std)
            z = dist.rsample().to(self.device)

            recon_images = self.decoder(z)


        # with torch.no_grad():
        #     pred, mean, std, _ = self.encoder(images)
        #
        #     print(mean.shape)
        #     print(std.shape)
        #
        #     # Get single samples from the distributions with reparametrisation trick
        #     dist = torch.distributions.normal.Normal(mean, std)
        #     z = dist.rsample().to(self.device)
        #
        #     recon_images = self.decoder(z)

        # return predictions and the loss
        # return recon_images

        return recon_images

    def interpolate(self, images: torch.Tensor, amount: int):
        with torch.no_grad():

            _, mean, std, _ = self.encoder(images)

            print(mean.shape)
            print(std.shape)

            mean_1, std_1 = mean[0,:], std[0,:]
            mean_2, std_2 = mean[1,:], std[1,:]

            all_mean  = torch.tensor([]).to(self.device)
            all_std = torch.tensor([]).to(self.device)

            diff_mean = mean_1 - mean_2
            diff_std = std_1 - std_2

            steps_mean = diff_mean / (amount-1)
            steps_std = diff_std / (amount-1)

            for i in range(amount):
                all_mean = torch.cat((all_mean, mean_1 - steps_mean*i))
                all_std = torch.cat((all_std, std_1 - steps_std*i))

            all_mean = all_mean.view(amount, -1)
            all_std = all_std.view(amount, -1)

            dist = torch.distributions.normal.Normal(all_mean, all_std)
            z = dist.rsample().to(self.device)

            recon_images = self.decoder(z)

        # with torch.no_grad():
        #     pred, mean, std, _ = self.encoder(images)
        #
        #     print(mean.shape)
        #     print(std.shape)
        #
        #     # Get single samples from the distributions with reparametrisation trick
        #     dist = torch.distributions.normal.Normal(mean, std)
        #     z = dist.rsample().to(self.device)
        #
        #     recon_images = self.decoder(z)

        # return predictions and the loss
        # return recon_images

        return recon_images

    def build_means(self, input: torch.Tensor):
        # print(f'build_means input: {input.shape}')
        _, mean, log_std, _ = self.encoder(input)

        self.means = torch.cat((self.means, mean))

        return


    def build_histo(self, input: torch.Tensor):
        """
            Creates histos or samples Qs from it
            NOTE:
            Make sure you only put faces into this
            functions
        """
        _, mean, std, _ = self.encoder(input)

        self.means = torch.cat((self.means, mean))
        self.std = torch.cat((self.std, std))

        values = norm.pdf(self.xlin, mean.unsqueeze(-1).cpu(), std.unsqueeze(-1).cpu()).sum(0)
        self.hist += values

        return

    def get_histo_max(self):
        probs = torch.zeros_like(self.means[:,0]).to(self.device)

        for i in range(self.z_dim):
            dist = self.means[:,i].cpu().numpy()

            hist, bins = np.histogram(dist, density=True, bins=self.num_bins)

            bins[0] = -float('inf')
            bins[-1] = float('inf')
            bin_idx = np.digitize(dist, bins)

            hist = hist + self.alpha
            hist /= np.sum(hist)

            p = 1.0/(hist[bin_idx-1])
            p /= np.sum(p)

            probs = torch.max(probs, torch.Tensor(p).to(self.device))

        probs /= probs.sum()

        return probs

    def get_histo_max5(self):
        probs = torch.zeros_like(self.means, dtype=float).to(self.device)

        for i in range(self.z_dim):
            dist = self.means[:,i].cpu().numpy()

            hist, bins = np.histogram(dist, density=True, bins=self.num_bins)

            bins[0] = -float('inf')
            bins[-1] = float('inf')
            bin_idx = np.digitize(dist, bins)

            hist = hist + self.alpha
            hist /= np.sum(hist)

            p = 1.0/(hist[bin_idx-1])
            p /= np.sum(p)

            probs[:,i] = torch.Tensor(p).to(self.device)

        probs = probs.sort(1, descending=True)[0][:,:5]
        probs = probs.prod(1)

        # print(probs)
        return probs

    def get_histo_max50(self):
        probs = torch.zeros_like(self.means, dtype=float).to(self.device)

        for i in range(self.z_dim):
            dist = self.means[:,i].cpu().numpy()

            hist, bins = np.histogram(dist, density=True, bins=self.num_bins)

            bins[0] = -float('inf')
            bins[-1] = float('inf')
            bin_idx = np.digitize(dist, bins)

            hist = hist + self.alpha
            hist /= np.sum(hist)

            p = 1.0/(hist[bin_idx-1])
            p /= np.sum(p)

            probs[:,i] = torch.Tensor(p).to(self.device)

        probs_idx = probs.argsort(1, descending=True)[:,:50][1] #getting indexes of most important variables for perturbing
        probs = probs.sort(1, descending=True)[0][:,:50]
        probs = probs.prod(1)

        print(f'probs_idx shape: {probs_idx.shape}')
        with open(f'results/logs/{self.config.test_no}/variable_idxs.pkl', 'wb') as f:
            pickle.dump(probs_idx, f)
        return probs

    def get_histo_gaussian(self):
        """
            Returns the probabilities given the means given the histo values
        """
        results = np.empty(self.means.shape[0])
        hist_batch_size = 4000
        # Iterate in large batches over dataset to prevent memory lockup
        for i in range(0, self.means.shape[0], hist_batch_size):
            i_end = i  + hist_batch_size
            if i_end > self.means.shape[0]:
                i_end = self.means.shape[0]
            mean = self.means[i:i_end, :]
            std = self.std[i:i_end, :]

            lins = norm.pdf(self.xlin, mean.unsqueeze(-1).cpu(), std.unsqueeze(-1).cpu())
            Q = lins * self.hist
            Q = Q.sum(-1)
            W = 1 / (Q + self.alpha)
            # Performing the max value technique, TODO: analyse top 5
            results[i:i_end] = W.max(-1)

        # # Reset values
        self.hist.fill(0)
        self.means = torch.Tensor().to(self.device)
        self.std = torch.Tensor().to(self.device)
        return torch.tensor(results).to(self.device)

    def recon_images(self, images: torch.Tensor):
        with torch.no_grad():
            pred, mean, std, _ = self.encoder(images)

            print(mean.shape)
            print(std.shape)

            # Get single samples from the distributions with reparametrisation trick
            dist = torch.distributions.normal.Normal(mean, std)
            z = dist.rsample().to(self.device)

            recon_images = self.decoder(z)

        # return predictions and the loss
        return recon_images

    def recon_images_perturb(self, images: torch.Tensor, var_to_perturb):
        with torch.no_grad():
            pred, mean, std, _ = self.encoder(images)

            perturbation_range = self.config.perturbation_range
            var_to_perturb = var_to_perturb #self.config.var_to_perturb
            recon_images = []
            for p in perturbation_range:
                mean[:,var_to_perturb] = mean[:,var_to_perturb] * p
                std[:, var_to_perturb] = std[:, var_to_perturb] * p
                # Get single samples from the distributions with reparametrisation trick

                dist = torch.distributions.normal.Normal(mean, std)
                z = dist.rsample().squeeze().to(self.device)
                # print(z[55])
                # z[var_to_perturb] = z[var_to_perturb] + p #adding perturbation value to specified latent variable
                recon_image = self.decoder(z.unsqueeze(0))
                recon_images.append(recon_image)

        # return predictions and the loss
        return recon_images

    def sample(self, n_samples, z_samples=[]):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        with torch.no_grad():
            z_samples = torch.randn(n_samples, self.z_dim).to(self.device)
            sampled_images = self.decoder(z_samples)

        return sampled_images
