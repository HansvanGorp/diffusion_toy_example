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

# %% function get true score
def get_true_score(x_in):
    x_in = nn.Parameter(x_in)
    
    log_p_mode1 = -torch.mean((x_in-locations[0,:])**2)
    log_p_mode2 = -torch.mean((x_in-locations[1,:])**2)
    
    log_p = torch.log(weight_mode1*torch.exp(log_p_mode1)+(1-weight_mode1)*torch.exp(log_p_mode2))
    
    log_p.backward()
    
    dx = x_in.grad
    
    return dx

# %% seed
torch.random.manual_seed(0)

# %% data and its parameters
X = torch.load("data.tar")

# %% parameters
weight_mode1 = 0.9 #weight_mode2 is inferred as 1-weight_mode1
locations =  torch.Tensor([[4.0,4],[-4,-4]])

# %% calculate the true score
nr_linspace_samples = 20
linspace = torch.linspace(-6.5,6.5,nr_linspace_samples)
_X,_Y = torch.meshgrid(linspace, linspace)

x = torch.vstack((_X.flatten(),_Y.flatten())).transpose(1,0)

gradients = torch.zeros_like(x)
for i in range(nr_linspace_samples**2):
    gradients[i,:]  = get_true_score(x[i,:])

# plot
Plot.generic_plot()
plt.quiver(x[:,0],x[:,1],gradients[:,0],gradients[:,1])
plt.savefig("figures//true_score.png",dpi=300,bbox_inches="tight")

x_quiver = x

# %% using the true score with gradient ascent
nr_samples = 10
T = 200
alpha = 1e-2

use_langevin = True


x_0 = -torch.randn(nr_samples,2)*4

x = torch.zeros(T,nr_samples,2)
x[0,:,:] = x_0

for t in range(T-1):
    score = torch.zeros(nr_samples,2)
    
    for i in range(nr_samples):
        score[i,:] = get_true_score(x[t,i,:])
    
    # update rule
    x[t+1,:,:] = x[t,:,:] + 0.5*alpha * score
    if use_langevin:
        z = torch.randn_like(x[0,:,:])
        x[t+1,:,:] = x[t+1,:,:] + alpha**0.5 * z
    
# %% make video 
# figures   
Path('video_figures').mkdir(parents=True, exist_ok=True)
time.sleep(1)


for t in range(T):
    Plot.generic_plot()
    plt.quiver(x_quiver[:,0],x_quiver[:,1],gradients[:,0],gradients[:,1])
    plt.scatter(x[t,:,0],x[t,:,1])
    plt.savefig(f"video_figures//{t}.png",dpi=300,bbox_inches="tight")
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
out = cv2.VideoWriter('videos//score_gradient_ascent.mp4', fourcc,24, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# delete the folder
dirpath = Path('video_figures')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
