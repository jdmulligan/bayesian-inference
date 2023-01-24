#!/usr/bin/env python

import os
import sys
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict

import common_base

####################################################################################################################
class DataIO(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, **kwargs):
        super(DataIO, self).__init__(**kwargs)

    #---------------------------------------------------------------
    # Write nested dictionary of ndarray to hdf5 file
    # Note: all keys should be strings
    #---------------------------------------------------------------
    def write_data(self, results, output_dir, filename):
        print()

        print(f'Writing results to {output_dir}/{filename}...')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dicttoh5(results, os.path.join(output_dir, filename), overwrite_data=True)

        print('Done.')
        print()

    #---------------------------------------------------------------
    # Read dictionary of ndarrays from hdf5
    # Note: all keys should be strings
    #---------------------------------------------------------------
    def read_data(self, input_file):
        print()
        print(f'Loading results from {input_file}...')

        results = h5todict(input_file)

        print('Done.')
        print()

        return results

    #---------------------------------------------------------------
    # Initialize from .dat files into a dictionary of numpy arrays
    #   - We loop through all observables in the table directory for the given model and parameterization
    #   - We include only those observables:
    #      - That have sqrts,centrality specified in the model_config
    #      - Whose filename contains a string from model_config observable_list
    #---------------------------------------------------------------
    def initialize_observables(self, table_dir, model_config, parameterization):
        print('Including the following observables:')
        observable_list_print = []
    
        # We will construct a dict containing all observables  
        observables = self.recursive_defaultdict()

        #----------------------
        # Read experimental data 
        data_dir = os.path.join(table_dir, 'Data')
        for filename in os.listdir(data_dir):
            if self.accept_observable(model_config, filename):

                data = np.loadtxt(os.path.join(data_dir, filename), ndmin=2)
                data_entry = {}
                data_entry['xmin'] = data[:,0]
                data_entry['xmax'] = data[:,1]
                data_entry['y'] = data[:,2]
                data_entry['y_err'] = data[:,3]

                sqrts, system, observable_type, observable, subobservable, centrality = self.filename_to_labels(filename)
                observable_list_print.append(f'  {sqrts}__{system}__{observable_type}__{observable}__{subobservable}__{centrality}')

                observables['Data'][sqrts][system][observable_type][observable][subobservable][centrality] = data_entry

                if 0 in data_entry['y']:
                    sys.exit(f'{filename} has value=0')

        #----------------------
        # Read design points 
        design_dir = os.path.join(table_dir, 'Design')
        for filename in os.listdir(design_dir):

            if self.filename_to_labels(filename) == parameterization: 
                observables['Design'] = np.loadtxt(os.path.join(design_dir, filename), ndmin=2)

        #----------------------
        # Read predictions and uncertainty
        prediction_dir = os.path.join(table_dir, 'Prediction')
        for filename_prediction in os.listdir(prediction_dir):

            if 'values' in filename and parameterization in filename:
                if self.accept_observable(model_config, filename):
                    filename_prediction_values = filename_prediction
                    filename_prediction_errors = filename_prediction.replace('values', 'errors') 

                    prediction_values = np.loadtxt(os.path.join(prediction_dir, filename_prediction_values), ndmin=2)
                    prediction_errors = np.loadtxt(os.path.join(prediction_dir, filename_prediction_errors), ndmin=2)

                    parameterization, sqrts, system, observable_type, observable, subobservable, centrality = self.filename_to_labels(filename_prediction_values)

                    observables['Prediction'][sqrts][system][observable_type][observable][subobservable][centrality]['y'] = prediction_values
                    observables['Prediction'][sqrts][system][observable_type][observable][subobservable][centrality]['y_err'] = prediction_errors

                    # TODO: Do something about bins that have value=0?
                    if 0 in prediction_values:
                        print(f'WARNING: {filename_prediction_values} has value=0 at design points {np.where(prediction_values == 0)[1]}')

                    # Check that data and prediction have same size
                    data_size = observables['Data'][sqrts][system][observable_type][observable][subobservable][centrality]['y'].shape[0]
                    prediction_size = observables['Data'][sqrts][system][observable_type][observable][subobservable][centrality]['y'].shape[0]
                    if data_size != prediction_size:
                        sys.exit(f'({filename_prediction}) has different shape ({prediction_size}) than Data ({data_size}).')

        #----------------------
        # Construct covariance matrices

        #----------------------
        # Print observables that we will use
        [print(s) for s in sorted(observable_list_print)]

        return observables

    #---------------------------------------------------------------
    # Parse filename
    #---------------------------------------------------------------
    def filename_to_labels(self, filename):

        filename_keys = filename[:-4].split('__')

        data_type = filename_keys[0]

        if data_type == 'Data':

            sqrts = filename_keys[1]
            system = filename_keys[2]
            observable_type = filename_keys[3]
            observable = filename_keys[4]
            subobserable = filename_keys[5]
            centrality = filename_keys[6]
            return sqrts, system, observable_type, observable, subobserable, centrality

        elif data_type == 'Design':

            parameterization = filename_keys[1]
            return parameterization

        elif data_type == 'Prediction':

            parameterization = filename_keys[1]
            sqrts = filename_keys[2]
            system = filename_keys[3]
            observable_type = filename_keys[4]
            observable = filename_keys[5]
            subobserable = filename_keys[6]
            centrality = filename_keys[7]
            return parameterization, sqrts, system, observable_type, observable, subobserable, centrality

    #---------------------------------------------------------------
    # Check if observable should be included in the analysis.
    # It must:
    #  - Have sqrts,centrality specified in the model_config
    #  - Have a filename that contains a string from model_config observable_list
    #---------------------------------------------------------------
    def accept_observable(self, model_config, filename):

        if 'Data' in filename:
            sqrts, _, _, _, _, centrality = self.filename_to_labels(filename)
        elif 'Prediction' in filename:
            _, sqrts, _, _, _, _, centrality = self.filename_to_labels(filename)

        # Check sqrts
        if int(sqrts) not in model_config['sqrts_list']:
            return False

        # Check centrality
        accepted_centrality = False
        centrality_min, centrality_max = centrality.split('-')
        if int(centrality_min) >= model_config['centrality_range'][0]:
            if int(centrality_max) <= model_config['centrality_range'][1]:
                accepted_centrality = True
        if not accepted_centrality:
            return False

        # Check observable
        found_observable = False
        for observable_string in model_config['observable_list']:
            if observable_string in filename:
                found_observable = True
        if not found_observable:
            return False

        return True