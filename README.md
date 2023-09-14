# Bayesian inference of QCD transport properties

This repository contains analysis code to implement Bayesian inference of one of the key emergent quantities of quantum chromodynamics (QCD), the jet transverse diffusion coefficient $\hat{q}$.
The initial results were published in [arXiv:2102.11337](https://inspirehep.net/literature/1847995), and the the most recent results were presented at [Quark Matter 2023](https://indico.cern.ch/event/1139644/).

The end-to-end workflow consists of:
- Simulating a physics model $f(\theta)$ at a collection of design points $\theta$ using the [JETSCAPE](https://github.com/JETSCAPE/JETSCAPE) framework – requiring $\mathcal{O}(10M)$ CPU-hours.
- Using PCA to reduce the dimensionality of the feature space.
- Fitting Gaussian Processes to emulate the physics model at any $\theta$.
- Sampling the posterior $P(\theta|D)$ using MCMC, with a Gaussian likelihood constructed by comparing the emulated physics model $f(\theta)$ to published experimental measurements $D$ from the Large Hadron Collider (LHC) and the Relativistic Heavy Ion Collider (RHIC).

This results in a constraint on the transverse diffusion coefficient $\hat{q}$ describing a jet propagating through deconfined QCD matter.
![image](https://github.com/jdmulligan/bayesian-inference/assets/16219745/faac0d39-39ad-4acf-a898-91ec51d57a31)


## Running the data pipeline

The data pipeline consists of the following optional steps:
1. Read in the design points, predictions, experimental data, and uncertainties
2. Perform PCA and fit GP emulators
3. Run MCMC
4. Plot results and validation

The analysis is steered by the script `steer_analysis.py`, where you can specify which parts of the pipeline you want to run, along with a config file (e.g. 'jet_substructure.yaml').

The config files will specify which steps to run along with input/output paths for each step, where applicable.

### To run the analysis:
```
python steer_analysis.py -c ./config/jet_substructure.yaml
```


## Setup software environment – example on hiccup cluster
<details>
  <summary>Click for details</summary>
<br/> 

### Logon and allocate a node

You can either use the usual hiccup CPU nodes, or hiccupgpu (useful if slurm queue is busy).

#### hiccup GPU

Logon to hiccupgpu:
```
ssh <user>@hic.lbl.gov -p 1142
```

#### hiccup CPU

Logon to hiccup:
```
ssh <user>@hic.lbl.gov
```

First, request an interactive node from the slurm batch system:
   ```
   srun -N 1 -n 20 -t 2:00:00 -p quick --pty bash
   ``` 
   which requests 1 full node (20 cores) for 2 hours in the `quick` queue. You can choose the time and queue: you can use the `quick` partition for up to a 2 hour session, `std` for a 24 hour session, or `long` for a 72 hour session – but you will wait longer for the longer queues). 
Depending how busy the queue is, you may get the node instantly, or you may have to wait awhile.
When you’re done with your session, just type `exit`.
Please do not run anything but the lightest tests on the login node. If you are finding that you have to wait a long time, let us know and we can take a node out of the slurm queue and logon to it directly.

### Initialize environment
  
Now we need to initialize the environment: set the python version, and create a virtual environment for python packages.
We have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd bayesian-inference
./init.sh --install
```
  
On subsequent times, you don't need to pass the `install` flag:
```
cd bayesian-inference
./init.sh
```

Now we are ready to run our scripts.

   
</details>
