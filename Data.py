# %% imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# local
import Plot

# %% seed
torch.random.manual_seed(0)

# %% parameters
weight_mode1 = 0.75 #weight_mode2 is inferred as 1-weight_mode1
locations =  torch.Tensor([[4.0,4],[-4,-4]])
nr_samples = 1000

# %% create samples
mode = 1*(torch.rand(nr_samples) > weight_mode1)

X = torch.randn(nr_samples,2) + locations[mode,:]

# %% save this data
torch.save(X,"data.tar")

# %% plot them
Plot.generic_plot()
plt.scatter(X[:,0],X[:,1])
plt.savefig("figures//data.png",dpi=300,bbox_inches="tight")


# %% plot p(x)
nr_linspace_samples = 100
linspace = torch.linspace(-6.5,6.5,nr_linspace_samples)
_X,_Y = torch.meshgrid(linspace, linspace)

x = torch.vstack((_X.flatten(),_Y.flatten())).transpose(1,0)

log_p_mode1 = -torch.mean((x-locations[0,:])**2,dim=1)
log_p_mode2 = -torch.mean((x-locations[1,:])**2,dim=1)

log_p = torch.log(weight_mode1*torch.exp(log_p_mode1)+(1-weight_mode1)*torch.exp(log_p_mode2))

log_p = log_p.reshape(nr_linspace_samples,nr_linspace_samples)

Plot.generic_plot()
plt.scatter(X[:,0],X[:,1]) 
levels = -torch.arange(60).flip(0)
plt.contour(_X, _Y, log_p, levels)
plt.savefig("figures//data_contours.png",dpi=300,bbox_inches="tight")