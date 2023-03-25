import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model import VariationalAutoEncoder

# hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIM = 784
HID_DIM = 200
Z_DIM = 20

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR_RATE = 3e-4 # Karpathy Constant

SAVE_PATH = "model/mnist_vae"
USE_SAVED_MODEL = True

# Training Loop
def train(data, save_path=None):
    model = VariationalAutoEncoder(IMG_DIM, HID_DIM, Z_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(data))
        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], 28*28)
            x_reconstructed, mu, sig = model(x)

            # loss calculation
            rcl = loss_fn(x_reconstructed, x)
            kld = -torch.sum(1 + torch.log(sig.pow(2)) - mu.pow(2) -sig.pow(2)) # KL Divergence
            loss = rcl + kld

            # grad desc
            optim.zero_grad()
            loss.backward()
            optim.step()
            loop.set_postfix(epoch=epoch, loss=loss.item())

    if save_path:
        torch.save(model.state_dict(), f"{save_path}")
    return model

# generates num_images kinds of image
def inference(model, data, label, num_images, save_dir="outputs/"):
    x = None
    for img, l in data:
        if label == l:
            x = img.view(1, -1).to(DEVICE)
            break
    
    # creating output directory if it doesn't exist
    out_dir = os.path.join(save_dir, str(label))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    images = model.inference(x, num_images)
    print(f"\nGenerating {num_images} images for label={label}")
    for idx, img in enumerate(images):
        img = img.view(28, 28)
        plt.imsave(os.path.join(out_dir, f"{idx}.png"), img, cmap="gray")
    print(f"Outputs saved to {out_dir}")

if __name__ == "__main__":
    mnist_data = datasets.MNIST("dataset/", download=True, train=True, transform=transforms.ToTensor())
    data_loader = DataLoader(mnist_data, shuffle=True, batch_size=BATCH_SIZE)

    if USE_SAVED_MODEL:
        print("Loading Trained Model...")
        model = VariationalAutoEncoder(IMG_DIM, HID_DIM, Z_DIM).to(DEVICE)
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    else:
        print("Model Training...")
        model = train(data_loader, save_path=SAVE_PATH)

    for label in range(10):
        num_images = 10
        inference(model, mnist_data, label, num_images)