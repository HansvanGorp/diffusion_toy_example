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

# %% noising rules
def beta_nosing_rule(x, beta_t):
    z = torch.randn_like(x)
    
    mean_mult = (1-beta_t)**0.5
    std_mult  = beta_t**0.5
    
    x = mean_mult*x + std_mult*z
    
    return x 

# %% data
X = torch.load("data.tar")
X = X


# %% parameters
T = 200

alpha = 1-(np.arange(T)+1)/T
beta = np.zeros_like(alpha)

for t in range(1,T):
    beta[t] = 1 - alpha[t]/alpha[t-1]
    
# %% diffusion process
x = torch.zeros(T,X.size(0),2)
x[0,:,:] = X

for t in range(T-1):
    x[t+1,:,:] = beta_nosing_rule(x[t,:,:], beta[t])
    
# %% make video 
if True:
    # create folder   
    Path('video_figures').mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    
    # figures   
    for t in range(T):
        Plot.generic_plot()
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
    out = cv2.VideoWriter('videos//diffusion_process.mp4', fourcc,24, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    # delete the folder
    dirpath = Path('video_figures')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

# %% plot trajectory of one sample
sample_id = 1
alphas = 1-beta

Plot.generic_plot()
for t in range(1,T):
    plt.scatter(x[t,sample_id,0],x[t,sample_id,1], alpha = alphas[t], c = 'r')
plt.scatter(x[0,sample_id,0],x[0,sample_id,1], c = 'b')

plt.savefig("figures//diffusion_single_forward.png",dpi=300,bbox_inches="tight")
plt.show()