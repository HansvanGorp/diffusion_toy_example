# Introduction
This is some spaghetti code to train several different generative models on a simple toy example. See the powerpoint presentation for further details.

## Environment
To make use of the environment, first install [Anaconda](https://www.anaconda.com/). In anaconda prompt navigate to the parent directory and run:
```
conda env create -f environment.yml --prefix ./env
```
and then activate the environment:
```
conda activate ./env
```

## Data
The data is generated using a simple 2D bimodal Gaussian. The two modes are located at [2,2] and [-2,-2], with a weighting of 0.75 and 0.25, respectively. To create data.tar again, run:
```
Data.py
```

## VAE, GAN, and Flow
Very simple generative baselines have been implemented, which can be run using:
```
VAE.py
GAN.py
Flow.py
```

## Score
### True Score
We first show what the 'true' score is, which we only know because we defined the data distribution ourselves analytically. We then show how a Langevin update rule will result in (disproportional) samples from the distribution. This can be recreated by running:
```
True_Score.py
```

### Estimating Score via Noise Matching
We then show how we can estimate the score by performing noise matching at different noise scales. We then again use Langevin dynamics, but we now also loop over the different noise scales. This does lead to proportional samples. 
```
Score.py
```

## Diffusion
### Diffusion Process
We show what a diffusion process looks like on our toy data.
```
Diffusion_Process.py
```

### Estimating the Reverse Diffusion Process
We then estimate the reverse of this diffusion process by again performing noise matching. but now with a different noise level for each time step T. We then show how we can sample from the diffusion process. In the video we show both the current sample at time step t, as well as the current end-estimate x_0.
```
Diffusion.py
```
