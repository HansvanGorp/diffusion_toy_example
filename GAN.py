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
nr_epochs = 200
batch_size = 4
features = 32
latent_size = 4

# %% set up GAN
class Generator(nn.Module):
    def __init__(self, features, latent_size):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 2),
            )
        self.latent_size = latent_size
        
    def forward(self, batch_size):
        z = torch.randn(batch_size,self.latent_size)
        x = self.layers(z)
        return x

class Discriminator(nn.Module):
    def __init__(self, features):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 1),
            nn.Sigmoid(),
            )
        
    def forward(self, x):
        y = self.layers(x)
        return y
    
# create instances
generator = Generator(features, latent_size)
discriminator = Discriminator(features)
optimizer_generator = torch.optim.Adam(generator.parameters())
optimizer_discriminator = torch.optim.Adam(discriminator.parameters())

# %% train
nr_batches = int(X.size(0)//batch_size)

for i in tqdm(range(nr_epochs)):
    for j in range(nr_batches):
        x_real = X[j*batch_size:(j+1)*batch_size]
        
        # update generator
        optimizer_generator.zero_grad()
        x_fake = generator(batch_size)
        y_fake = discriminator(x_fake)
        
        bce_loss_generator = -torch.mean(torch.log(y_fake))
        bce_loss_generator.backward()
        optimizer_generator.step()
        
        # update discriminator
        optimizer_discriminator.zero_grad()
        
        y_fake = discriminator(x_fake.detach())
        y_real = discriminator(x_real)
        
        bce_loss_discriminator = -0.5*torch.mean(torch.log(y_real))-0.5*torch.mean(torch.log(1-y_fake))
        bce_loss_discriminator.backward()
        optimizer_discriminator.step()
        
# %% plot some examples
with torch.no_grad():
    X_fake = generator(100)

Plot.generic_plot()
plt.scatter(X[:,0],X[:,1])
plt.scatter(X_fake[:,0],X_fake[:,1])
plt.savefig("figures//GAN.png",dpi=300,bbox_inches="tight")
