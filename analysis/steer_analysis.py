#! /usr/bin/env python
'''
Main script to steer Bayesian inference studies for heavy-ion jet analysis

authors: J.Mulligan
Based in part on JETSCAPE/STAT code.
'''

import argparse
import os
import sys
import yaml

import common_base

import data_IO
import run_analysis

####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', **kwargs):

        # Initialize config file
        self.config_file = config_file
        self.initialize()

        # Create data IO class
        self.data_IO = data_IO.DataIO()

        print(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self):
        print('Initializing class objects')

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.output_dir = config['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']

        self.initialize_observables = config['initialize_observables']
        self.fit_emulators = config['fit_emulators']
        self.run_mcmc = config['run_mcmc']
        self.plot = config['plot']
    
        self.analyses = config['analyses']

    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):

        # Loop through each analysis
        for analysis_name,analysis_config in self.analyses.items():

            # Loop through the parameterizations
            for parameterization in analysis_config['parameterizations']:

                # Initialize design points, predictions, data, and uncertainties
                # We store them in a dict and write/read it to HDF5
                if self.initialize_observables:
                    print()
                    print(f'Initializing model: {analysis_name} ({parameterization} parameterization)...')

                    # We separate out the validation indices specified in the config
                    validation_range = analysis_config['validation_indices']
                    validation_indices = range(validation_range[0], validation_range[1])

                    observables = self.data_IO.initialize_observables(self.observable_table_dir, 
                                                                      analysis_config, 
                                                                      parameterization,
                                                                      validation_indices)
                    self.data_IO.write_data(observables, 
                                            os.path.join(self.output_dir, f'{analysis_name}_{parameterization}'), 
                                            filename='observables.h5')

                # Fit emulators
                if self.fit_emulators:

                    # Do PCA
                    continue

                # Run MCMC
                if self.run_mcmc:

                    analysis = run_analysis.RunAnalysis(config_file=self.config_file,
                                                        model=model,
                                                        output_dir=self.output_dir,
                                                        kfold_index=self.kfold_index)
                    analysis.initialize()
                    analysis.run_model()
                    
                # Plot   
                if self.plot:
                    
                    continue


####################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jet Bayesian Analysis')
    parser.add_argument('-c', '--configFile', 
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='../config/Analysis1.yaml', )
    args = parser.parse_args()

    print('Configuring...')
    print(f'  configFile: {args.configFile}')

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print(f'File {args.configFile} does not exist! Exiting!')
        sys.exit(0)

    analysis = SteerAnalysis(config_file=args.configFile)
    analysis.run_analysis()
