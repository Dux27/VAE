import torch
import numpy as np
import torch.nn as nn
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam

# create a transform to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
current_path = os.getcwd()
path = os.path.join(current_path, 'datasets')
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset  = MNIST(path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Used device: " + ("Cuda" if torch.cuda.is_available() else "CPU"))

class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 5)  # Number of parameters
        self.logvar_layer = nn.Linear(latent_dim, 5)  # Number of parameters
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(5, latent_dim),  # Number of parameters
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(device)
        z = mean + torch.exp(log_var/2) * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
    
model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)    

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, 784).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss

if __name__ == "__main__":
    train(model, optimizer, epochs=40, device=device)

    # Save the model parameters
    torch.save(model.state_dict(), os.path.join(current_path, 'vae_model_latent_200_2.pth'))
