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
model.load_state_dict(torch.load(os.path.join(current_path, 'weights\\vae_model_latent_200_5.pth'), weights_only=True))
model.eval()

def generate_digit(first, second, third, fourth, fifth):
    z_sample = torch.tensor([[first, second, third, fourth, fifth]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

def update_digit(event=None):
    first = first_slider.get()
    second = second_slider.get()
    third = third_slider.get()
    fourth = fourth_slider.get()
    fifth = fifth_slider.get()
    generate_digit(first, second, third, fourth, fifth)

# Create the main window
root = tk.Tk()
root.title("VAE Digit Generator")
root.geometry("300x500")

# Create sliders for mean and variance
tk.Label(root, text="Parameters", font=24).pack()

first_slider = Scale(root, from_=-3.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="First", command=update_digit, tickinterval=0)
first_slider.set(0.0)
first_slider.pack(fill=tk.X, expand=True)

second_slider = Scale(root, from_=-3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Second", command=update_digit, tickinterval=0)
second_slider.set(0.0)
second_slider.pack(fill=tk.X, expand=True)

third_slider = Scale(root, from_=-3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Third", command=update_digit, tickinterval=0)
third_slider.set(0.0)
third_slider.pack(fill=tk.X, expand=True)

fourth_slider = Scale(root, from_=-3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Fourth", command=update_digit, tickinterval=0)
fourth_slider.set(0.0)
fourth_slider.pack(fill=tk.X, expand=True)

fifth_slider = Scale(root, from_=-3, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Fifth", command=update_digit, tickinterval=0)
fifth_slider.set(0.0)
fifth_slider.pack(fill=tk.X, expand=True)

# Run the GUI event loop
root.mainloop()
