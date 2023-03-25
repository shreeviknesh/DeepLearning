import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    # img -> hid -> [mu, sigma -> reparametrization] -> hid -> img
    def __init__(self, img_dim, hid_dim=200, z_dim=20):
        super().__init__()
        self.img_hid = nn.Linear(img_dim, hid_dim)
        self.hid_mu = nn.Linear(hid_dim, z_dim)
        self.hid_sig = nn.Linear(hid_dim, z_dim)

        self.z_hid = nn.Linear(z_dim, hid_dim)
        self.hid_img = nn.Linear(hid_dim, img_dim)

        self.relu = nn.ReLU()

    # img -> hid -> mu, sigma
    def encode(self, x):
        hid = self.relu(self.img_hid(x))
        mu, sig = self.hid_mu(hid), self.hid_sig(hid)
        return mu, sig

    # mu, sigma -> hid -> img
    def decode(self, z):
        hid = self.relu(self.z_hid(z))
        img = torch.sigmoid(self.hid_img(hid))
        return img

    def forward(self, x):
        mu, sig = self.encode(x)
        epsilon = torch.randn_like(sig)
        z_reparametrized = mu + (sig * epsilon) # reparametrization
        img_reconstructed = self.decode(z_reparametrized)
        return img_reconstructed

    # helper function to reconstruct new images from image
    @torch.no_grad()
    def inference(self, x, num_imgs):
        images = []
        for _ in range(num_imgs):
            mu, sig = self.encode(x)
            epsilon = torch.randn_like(sig)
            z_reparametrized = mu + sig * epsilon
            img_reconstructed = self.decode(z_reparametrized).cpu()
            images.append(img_reconstructed)
        return images