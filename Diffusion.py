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

# %% noising rules
def alpha_nosing_rule(x, alpha_t):
    z = torch.randn_like(x)
    
    mean_mult = (alpha_t)**0.5
    std_mult  = (1-alpha_t)**0.5
    
    x = mean_mult*x + std_mult*z
    
    return x 

def calcualte_gradient(denoiser,x_quiver,alpha,t):
    with torch.no_grad():
        x_new  = alpha[t-1]**0.5*denoiser(x_quiver,alpha[t])
    return x_new-x_quiver

# %% parameters
nr_epochs = 1000
batch_size = 8
features = 32

T = 500

alpha = 1-(np.arange(T)+1)/T

# %% network
class Denoiser(nn.Module):
    def __init__(self, features):
        super(Denoiser, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, 2),
            )
        
    def forward(self, x, alpha_t):
        alpha_t = torch.tensor([alpha_t],dtype = x.dtype)
        alpha_t = alpha_t.repeat(x.size(0)).unsqueeze(1)
        x = torch.hstack((x, alpha_t))
        x = self.layers(x)
        return x
    
# create instances
denoiser = Denoiser(features)
optimizer = torch.optim.Adam(denoiser.parameters())

# %% train
nr_batches = int(X.size(0)//batch_size)

for i in tqdm(range(nr_epochs)):
    for j in range(nr_batches):
        x = X[j*batch_size:(j+1)*batch_size]
        
        optimizer.zero_grad()
        t = np.random.randint(0,high=T)
        
        x_noisy = alpha_nosing_rule(x, alpha[t])
        
        x_hat = denoiser(x_noisy, alpha[t])
        
        loss = torch.mean((x_hat - x)**2)
        
        loss.backward()
        optimizer.step()
        
# %% sampling process
nr_samples = 100

x_T = torch.randn(nr_samples,2)

x_all = torch.zeros(T , nr_samples, 2)
x_hat_all = torch.zeros(T , nr_samples, 2)
x_all[-1,:,:] = x_T
x_hat_all[-1,:,:] = x_T

for t in reversed(range(1,T)):
    x_now = x_all[t,:,:]
    
    with torch.no_grad():
        x_hat = denoiser(x_now, alpha[t])
    
    x_next = alpha_nosing_rule(x_hat, alpha[t-1])
    
    x_hat_all[t-1,:,:] = x_hat
    x_all[t-1,:,:] = x_next
    
# %% make video 
nr_linspace_samples = 20
linspace = torch.linspace(-6.5,6.5,nr_linspace_samples)
_X,_Y = torch.meshgrid(linspace, linspace)
x_quiver = torch.vstack((_X.flatten(),_Y.flatten())).transpose(1,0)

if True:
    # create folder   
    Path('video_figures').mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    
    # figures   
    for t in range(1,T-1):
        s = calcualte_gradient(denoiser,x_quiver,alpha,t)
        
        plt.figure(figsize = (10,5))
        
        plt.subplot(1,2,1)
        plt.scatter(x_all[t,:,0],x_all[t,:,1])
        plt.quiver(x_quiver[:,0],x_quiver[:,1],s[:,0],s[:,1])
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-6.5,6.5)
        plt.ylim(-6.5,6.5)
        plt.title("x_t")
        
        plt.subplot(1,2,2)
        plt.scatter(x_hat_all[t,:,0],x_hat_all[t,:,1])
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-6.5,6.5)
        plt.ylim(-6.5,6.5)
        plt.title("x_0")
        
        plt.savefig(f"video_figures//{T-t}.png",dpi=300,bbox_inches="tight")
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
    out = cv2.VideoWriter('videos//Diffusion.mp4', fourcc, 12, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    # delete the folder
    dirpath = Path('video_figures')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

