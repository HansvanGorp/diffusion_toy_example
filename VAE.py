# %% imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# local
import Plot

# %% seed
torch.random.manual_seed(0)

# %% data
X = torch.load("data.tar")

# %% parameters
nr_epochs = 100
batch_size = 4
features = 32
latent_size = 2

# %% set up GAN
class Encoder(nn.Module):
    def __init__(self, features, latent_size):
        super(Encoder, self).__init__()
        self.mu_layers = nn.Sequential(
            nn.Linear(2, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, latent_size),
            )
        
        self.sigma_layers = nn.Sequential(
            nn.Linear(2, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, latent_size),
            )
        
    def forward(self, x):
        mu = self.mu_layers(x)
        log_var = self.sigma_layers(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, features, latent_size):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 2)
            )
        
    def forward(self, x):
        y = self.layers(x)
        return y
    
class VAE(nn.Module):
    def __init__(self, features, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(features, latent_size)
        self.decoder = Decoder(features, latent_size)
        self.latent_size = latent_size
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5*log_var) * eps
        
        x_hat = self.decoder(z)
        
        return x_hat, mu, log_var
    
    def generate(self, nr_samples):
        z = torch.randn(nr_samples,self.latent_size)
        x = self.decoder(z)
        return x
    
# create instances
vae = VAE(features, latent_size)
optimizer = torch.optim.Adam(vae.parameters())

# %% train
nr_batches = int(X.size(0)//batch_size)

for i in tqdm(range(nr_epochs)):
    for j in range(nr_batches):
        x = X[j*batch_size:(j+1)*batch_size]
        
        optimizer.zero_grad()
        x_hat, mu, log_var = vae(x)
        
        mse_loss = torch.mean((x_hat-x)**2)
        KL_loss  = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var),dim=1),dim=0)
        elbo = mse_loss + KL_loss
        
        elbo.backward()
        optimizer.step()
        
# %% plot some examples
with torch.no_grad():
    X_fake = vae.generate(100)

Plot.generic_plot()
plt.scatter(X[:,0],X[:,1])
plt.scatter(X_fake[:,0],X_fake[:,1])
plt.savefig("figures//VAE.png",dpi=300,bbox_inches="tight")


# %% ELBO of 
nr_linspace_samples = 500
linspace = torch.linspace(-6.5,6.5,nr_linspace_samples)
X,Y = torch.meshgrid(linspace, linspace)
x = torch.vstack((X.flatten(),Y.flatten())).transpose(1,0)

with torch.no_grad():
    x_hat, mu, log_var = vae(x)

mse_loss = torch.mean((x_hat-x)**2, dim = 1)
KL_loss  = -0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var),dim=1)
elbo = mse_loss + KL_loss

elbo = elbo.reshape(nr_linspace_samples,nr_linspace_samples)

plt.figure()
plt.imshow(-torch.log(elbo),origin='lower')
plt.xticks([])
plt.yticks([])
plt.savefig("figures//VAE_ELBO.png",dpi=300,bbox_inches="tight")