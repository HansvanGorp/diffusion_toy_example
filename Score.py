# %% imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
import shutil
import cv2
import time
from glob import glob
from natsort import natsorted

# local
import Plot

# %% seed
torch.random.manual_seed(0)

# %% data and its parameters
X = torch.load("data.tar")

# %% parameters
nr_epochs = 1000
batch_size = 4
features = 64
sigma_at_level  = torch.Tensor([0.25,1.0,4.0])
L = len(sigma_at_level)

# %% network
class Score(nn.Module):
    def __init__(self, features):
        super(Score, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 2),
            )
        
    def forward(self, x, sigma):
        sigma = torch.tensor([sigma])
        sigma = sigma.repeat(x.size(0)).unsqueeze(1)
        x = torch.hstack((x, sigma))
        x = self.layers(x)
        return x
    
# create instances
score = Score(features)
optimizer = torch.optim.Adam(score.parameters())

# %% train
nr_batches = int(X.size(0)//batch_size)

for i in tqdm(range(nr_epochs)):
    for j in range(nr_batches):
        x = X[j*batch_size:(j+1)*batch_size]
        
        optimizer.zero_grad()
        sigma_idx = np.random.randint(0,high=L)
        sigma = sigma_at_level[sigma_idx]
        
        z = torch.randn_like(x)
        x_noisy = x + z*sigma**2
        
        s = score(x_noisy, sigma)
        
        loss = torch.mean((s + z)**2)
        
        loss.backward()
        optimizer.step()


# %% plot this
nr_linspace_samples = 20
linspace = torch.linspace(-6.5,6.5,nr_linspace_samples)
_X,_Y = torch.meshgrid(linspace, linspace)

x_quiver = torch.vstack((_X.flatten(),_Y.flatten())).transpose(1,0)

for l in range(L):
    sigma = sigma_at_level[l]
    with torch.no_grad():   
        s = score(x_quiver, sigma)
        
    Plot.generic_plot()
    plt.quiver(x_quiver[:,0],x_quiver[:,1],s[:,0],s[:,1])
    plt.savefig(f"figures//estimated_score_{l}.png",dpi=300,bbox_inches="tight")
    plt.close()
    
# %% plot what data looks like for each of the noise levels:
for l in range(L):
    sigma = sigma_at_level[l]
    z = torch.randn_like(X)
    X_noisy = X + z*sigma**2
        
    Plot.generic_plot()
    plt.scatter(X[:,0],X[:,1])
    plt.scatter(X_noisy[:,0],X_noisy[:,1])
    plt.savefig(f"figures//data_score_{l}.png",dpi=300,bbox_inches="tight")
    plt.close()
    
# %% langevin dynamics
nr_samples = 100
T = 200
alpha = 4e-3

x_0 = -torch.randn(nr_samples,2)*4

x = torch.zeros(L, T , nr_samples, 2)
x[0,0,:,:] = x_0

for l, sigma in enumerate(sigma_at_level.flip(0)):
    # new lr
    alpha_l = alpha * (sigma**2)/(sigma_at_level[0]**2)
    
    # copy over if not first
    if l != 0 :
        x[l,0,:,:] = x[l-1,-1,:,:]
    
    # go through T steps
    for t in range(T-1):
        with torch.no_grad():
            s = score(x[l,t,:,:], sigma)
        
        # update rule
        z = torch.randn_like(x[0,0,:,:])
        x[l,t+1,:,:] = x[l,t,:,:] + 0.5*alpha_l * s + alpha_l**0.5 * z

# end result
for l in range(L):
    Plot.generic_plot()
    plt.scatter(x[l,-1,:,0],x[l,-1,:,1])
    plt.show()

# %% make video 
if True:
    # create folder   
    Path('video_figures').mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    
    # figures   
    for l, sigma in enumerate(sigma_at_level.flip(0)):
        with torch.no_grad():   
            s = score(x_quiver, sigma)
        
        for t in range(T):
            Plot.generic_plot()
            plt.quiver(x_quiver[:,0],x_quiver[:,1],s[:,0],s[:,1])
            plt.scatter(x[l,t,:,0],x[l,t,:,1])
            plt.savefig(f"video_figures//{l}_{t}.png",dpi=300,bbox_inches="tight")
            plt.close()
        
    # video
    img_array = []
    file_names = glob('video_figures//*.png')
    file_names = natsorted(file_names)
    
    for j in range(len(file_names)):
        img = cv2.imread(file_names[j])
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('videos//Score.mp4', fourcc,24, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    # delete the folder
    dirpath = Path('video_figures')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
