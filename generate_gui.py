import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train import VAE
import tkinter as tk
from tkinter import Scale

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

def update_digit(event=None):
    mean = mean_slider.get()
    var = var_slider.get()
    generate_digit(mean, var)

# Create the main window
root = tk.Tk()
root.title("VAE Digit Generator")
root.geometry("300x200")

# Create sliders for mean and variance
mean_slider = Scale(root, from_=-3.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Mean", command=update_digit, tickinterval=0)
mean_slider.set(0.0)
mean_slider.pack(fill=tk.X, expand=True)

var_slider = Scale(root, from_=-3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Variance", command=update_digit, tickinterval=0)
var_slider.set(0.0)
var_slider.pack(fill=tk.X, expand=True)

# Run the GUI event loop
root.mainloop()
