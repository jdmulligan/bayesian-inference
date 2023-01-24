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
    #
    # Return a dictionary with the following structure:
    #    observables['Data'][observable_label]['y'] -- value
    #                                         ['y_err'] -- total uncertainty (TODO: include uncertainty breakdowns)
    #                                         ['xmin'] -- bin lower edge (used only for plotting)
    #                                         ['xmax'] -- bin upper edge (used only for plotting)
    #    observables['Design'][parameterization] -- design points for a given parameterization
    #    observables['Prediction'][observable_label]['y'] -- value
    #                                               ['y_err'] -- statistical uncertainty
    #
    # where observable_label follows the convention from the table filenames:
    #      observable_label = f'{sqrts}__{system}__{observable_type}__{observable}__{subobservable}__{centrality}'
    #
    # Note that all of the data points are the ratio of AA/pp
    #---------------------------------------------------------------
    def initialize_observables(self, table_dir, model_config, parameterization):
        print('Including the following observables:')
    
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

                observable_label, _ = self.filename_to_labels(filename)
                observables['Data'][observable_label] = data_entry

                if 0 in data_entry['y']:
                    sys.exit(f'{filename} has value=0')

        #----------------------
        # Read design points 
        design_dir = os.path.join(table_dir, 'Design')
        for filename in os.listdir(design_dir):

            if self.filename_to_labels(filename)[1] == parameterization: 
                observables['Design'] = np.loadtxt(os.path.join(design_dir, filename), ndmin=2)

        #----------------------
        # Read predictions and uncertainty
        prediction_dir = os.path.join(table_dir, 'Prediction')
        for filename in os.listdir(prediction_dir):

            if 'values' in filename and parameterization in filename:
                if self.accept_observable(model_config, filename):

                    filename_prediction_values = filename
                    filename_prediction_errors = filename.replace('values', 'errors') 

                    prediction_values = np.loadtxt(os.path.join(prediction_dir, filename_prediction_values), ndmin=2)
                    prediction_errors = np.loadtxt(os.path.join(prediction_dir, filename_prediction_errors), ndmin=2)

                    observable_label, _ = self.filename_to_labels(filename_prediction_values)

                    observables['Prediction'][observable_label]['y'] = prediction_values
                    observables['Prediction'][observable_label]['y_err'] = prediction_errors

                    # TODO: Do something about bins that have value=0?
                    if 0 in prediction_values:
                        print(f'WARNING: {filename_prediction_values} has value=0 at design points {np.where(prediction_values == 0)[1]}')

                    # Check that data and prediction have same observables with the same size
                    if observable_label not in observables['Data']:
                        data_keys = observables['Data'].keys()
                        sys.exit(f'{observable_label} not found in observables[Data]: {data_keys}')
                    
                    data_size = observables['Data'][observable_label]['y'].shape[0]
                    prediction_size = observables['Prediction'][observable_label]['y'].shape[0]
                    if data_size != prediction_size:
                        sys.exit(f'({filename_prediction_values}) has different shape ({prediction_size}) than Data ({data_size}).')

        #----------------------
        # Construct covariance matrices

        #----------------------
        # Print observables that we will use
        [print(f'  {s}') for s in sorted(observables['Prediction'].keys())]

        return observables

    #---------------------------------------------------------------
    # Parse filename to return observable_label, parameterization
    #---------------------------------------------------------------
    def filename_to_labels(self, filename):

        # Remove file suffix
        filename_keys = filename[:-4].split('__')

        # Get table type and return observable_label, parameterization
        data_type = filename_keys[0]

        if data_type == 'Data':

            observable_label = '__'.join(filename_keys[1:])
            parameterization = None

        elif data_type == 'Design':

            observable_label = None
            parameterization = filename_keys[1]

        elif data_type == 'Prediction':

            parameterization = filename_keys[1]
            observable_label = '__'.join(filename_keys[2:-1])
        
        return observable_label, parameterization

    #---------------------------------------------------------------
    # Parse filename into individual keys
    #---------------------------------------------------------------
    def observable_label_to_keys(self, observable_label):

        observable_keys = observable_label.split('__')

        sqrts = observable_keys[0]
        system = observable_keys[1]
        observable_type = observable_keys[2]
        observable = observable_keys[3]
        subobserable = observable_keys[4]
        centrality = observable_keys[5]
        return sqrts, system, observable_type, observable, subobserable, centrality

    #---------------------------------------------------------------
    # Check if observable should be included in the analysis.
    # It must:
    #  - Have sqrts,centrality specified in the model_config
    #  - Have a filename that contains a string from model_config observable_list
    #---------------------------------------------------------------
    def accept_observable(self, model_config, filename):

        observable_label, _ = self.filename_to_labels(filename)

        sqrts, _, _, _, _, centrality = self.observable_label_to_keys(observable_label)

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