# bayesian-inference

This is a playground to implement Bayesian inference for heavy-ion jet measurements.

## Setup software environment – on hiccup cluster
<details>
  <summary>Click for details</summary>
<br/> 
  
### Logon and allocate a node
  
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

## Run the data pipeline

The data pipeline consists of the following optional steps:
1. Read in the design points, predictions, experimental data, and uncertainties
2. Fit emulators
3. Run MCMC
4. Plot

The analysis is steered by the script `steer_analysis.py`, where you can specify which parts of the pipeline you want to run, along with a config file (e.g. `hadron_jet_RAA.yaml` or 'substructure.yaml').

The config files will specify which steps to run along with input/output paths for each step, where applicable.

### To run the analysis:
```
python steer_analysis.py -c ./config/hadron_jet_RAA.yaml
```