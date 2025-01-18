import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train import VAE

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)

# Load the model parameters
current_path = os.getcwd()
model.load_state_dict(torch.load(os.path.join(current_path, 'vae_model.pth'), weights_only=True))
model.eval()

def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

generate_digit(0.0, 1.0)
generate_digit(1.0, 0.0)
