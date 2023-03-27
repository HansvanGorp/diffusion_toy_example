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

# %% set up GAN
class CouplingNet(nn.Module):
    def __init__(self, features):
        super(CouplingNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 2),
            )
        
    def forward(self, x):
        s_t = self.layers(x.unsqueeze(1))
        s = s_t[:,0]
        t = s_t[:,1]
        return s,t

class AffineLayer(nn.Module):
    def __init__(self, features):
        super(AffineLayer, self).__init__()
        self.coupling_net = CouplingNet(features)
        
    def forward(self, x, logdet):
        x1 = x[:,0]
        x2 = x[:,1]
        
        s,t = self.coupling_net(x2)
        
        x1 = x1*s + t
        
        x = torch.vstack((x1,x2)).transpose(1,0)
        
        logdet += torch.log(torch.abs(s))
        
        return x, logdet
    
    def reverse(self,x):
        x1 = x[:,0]
        x2 = x[:,1]
        
        s,t = self.coupling_net(x2)
        
        x1 = (x1-t)/s
        
        x = torch.vstack((x1,x2)).transpose(1,0)
        
        return x
    
class Flow(nn.Module):
    def __init__(self, features):
        super(Flow, self).__init__()
        self.affine_layer_1 = AffineLayer(features)
        self.affine_layer_2 = AffineLayer(features)
        self.affine_layer_3 = AffineLayer(features)
        self.affine_layer_4 = AffineLayer(features)
        
    def forward(self, x):
        logdet = 0
        x, logdet = self.affine_layer_1(x,logdet)
        x = x.flip(1)
        x, logdet = self.affine_layer_2(x,logdet)
        x = x.flip(1)
        x, logdet = self.affine_layer_3(x,logdet)
        x = x.flip(1)
        x, logdet = self.affine_layer_4(x,logdet)
        x = x.flip(1)
        
        return x, logdet
    
    def generate(self, nr_samples):
        x = torch.randn(nr_samples,2)*0.5
        
        x = x.flip(1)
        x = self.affine_layer_4.reverse(x)
        x = x.flip(1)
        x = self.affine_layer_3.reverse(x)    
        x = x.flip(1)
        x = self.affine_layer_2.reverse(x)    
        x = x.flip(1)
        x = self.affine_layer_1.reverse(x)    

        return x
    
# create instances
flow = Flow(features)
optimizer = torch.optim.Adam(flow.parameters())

# %% train
nr_batches = int(X.size(0)//batch_size)

for i in tqdm(range(nr_epochs)):
    for j in range(nr_batches):
        x = X[j*batch_size:(j+1)*batch_size]
        
        optimizer.zero_grad()
        z, logdet = flow(x)
        
        loss = torch.mean((z)**2) - torch.mean(logdet)
        
        loss.backward()
        optimizer.step()
        
# %% plot some examples
with torch.no_grad():
    X_fake = flow.generate(100)

Plot.generic_plot()
plt.scatter(X[:,0],X[:,1])
plt.scatter(X_fake[:,0],X_fake[:,1])
plt.savefig("figures//Flow.png",dpi=300,bbox_inches="tight")


# %% NLL of 
nr_linspace_samples = 500
linspace = torch.linspace(-6.5,6.5,nr_linspace_samples)
X,Y = torch.meshgrid(linspace, linspace)
x = torch.vstack((X.flatten(),Y.flatten())).transpose(1,0)

with torch.no_grad():
    z, logdet = flow(x)

nll = torch.mean((z)**2,dim=1) - logdet
nll = nll.reshape(nr_linspace_samples,nr_linspace_samples)

plt.figure()
plt.imshow(torch.exp(-nll),origin='lower')
plt.xticks([])
plt.yticks([])
plt.savefig("figures//Flow_nll.png",dpi=300,bbox_inches="tight")