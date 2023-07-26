#! /usr/bin/env python
'''
Main script to steer Bayesian inference studies for heavy-ion jet analysis

authors: J.Mulligan
Based in part on JETSCAPE/STAT code.
'''

import argparse
import logging
import os
import yaml

from bayesian_inference import data_IO
from bayesian_inference import emulation
from bayesian_inference import mcmc
from bayesian_inference import plot_emulation
from bayesian_inference import plot_mcmc

from bayesian_inference import common_base, helpers

logger = logging.getLogger(__name__)


####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', **kwargs):

        # Initialize config file
        self.config_file = config_file
        self.initialize()

        logger.info(self)

    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self):
        logger.info('Initializing class objects')

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
                    logger.info("")
                    logger.info('========================================================================')
                    logger.info(f'Initializing model: {analysis_name} ({parameterization} parameterization)...')
                    observables = data_IO.initialize_observables_dict_from_tables(self.observable_table_dir,
                                                                                  analysis_config,
                                                                                  parameterization)
                    data_IO.write_dict_to_h5(observables,
                                             os.path.join(self.output_dir, f'{analysis_name}_{parameterization}'),
                                             filename='observables.h5')

                # Fit emulators and write them to file
                if self.fit_emulators:
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Fitting emulators for {analysis_name}_{parameterization}...')
                    emulation_config = emulation.EmulationConfig(analysis_name=analysis_name,
                                                                 parameterization=parameterization,
                                                                 analysis_config=analysis_config,
                                                                 config_file=self.config_file)
                    emulation.fit_emulators(emulation_config)

                # Run MCMC
                if self.run_mcmc:
                    logger.info("")
                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Running MCMC for {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                  parameterization=parameterization,
                                                  analysis_config=analysis_config,
                                                  config_file=self.config_file)
                    mcmc.run_mcmc(mcmc_config)

        # Plot
        if self.plot:

            # Plots for individual analysis
            for analysis_name,analysis_config in self.analyses.items():
                for parameterization in analysis_config['parameterizations']:

                    logger.info('========================================================================')
                    logger.info(f'Plotting for {analysis_name} ({parameterization} parameterization)...')
                    logger.info("")

                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting emulators for {analysis_name}_{parameterization}...')
                    emulation_config = emulation.EmulationConfig(analysis_name=analysis_name,
                                                                 parameterization=parameterization,
                                                                 analysis_config=analysis_config,
                                                                 config_file=self.config_file)
                    plot_emulation.plot(emulation_config)
                    logger.info(f'Done!')
                    logger.info("")

                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting MCMC for {analysis_name}_{parameterization}...')
                    mcmc_config = mcmc.MCMCConfig(analysis_name=analysis_name,
                                                  parameterization=parameterization,
                                                  analysis_config=analysis_config,
                                                  config_file=self.config_file)
                    plot_mcmc.plot(mcmc_config)
                    logger.info(f'Done!')
                    logger.info("")

                    logger.info('------------------------------------------------------------------------')
                    logger.info(f'Plotting qhat results {analysis_name}_{parameterization}...')
                    logger.info(f'Done!')
                    logger.info("")

            # Plots across multiple analyses


####################################################################################################################
if __name__ == '__main__':
    helpers.setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Jet Bayesian Analysis')
    parser.add_argument('-c', '--configFile',
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='../config/hadron_jet_RAA.yaml', )
    args = parser.parse_args()

    logger.info('Configuring...')
    logger.info(f'  configFile: {args.configFile}')

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        msg = f'File {args.configFile} does not exist! Exiting!'
        logger.info(msg)
        raise ValueError(msg)

    analysis = SteerAnalysis(config_file=args.configFile)
    analysis.run_analysis()
