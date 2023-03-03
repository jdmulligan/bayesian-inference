#!/usr/bin/env python
'''
Module related to reading and writing of tables of observables into numpy arrays

The main functionalities are:
 - initialize_observables_dict_from_tables() -- read design/prediction/data tables (.dat files) into nested dictionary of numpy arrays
 - read/write_dict_to_h5() -- read/write nested dict of numpy arrays to HDF5  
 - predictions_matrix_from_h5() -- construct prediction matrix (design_points, observable_bins) from observables.h5 
 - design_array_from_h5() -- read design points from observables.h5 
 - observable_label_to_keys() -- convert observable string label to list of subobservables strings
 - sorted_observable_list_from_dict() -- get sorted list of observable_label keys, using fixed ordering convention that we enforce

authors: J.Mulligan, R.Ehlers
'''

import os
import sys
from collections import defaultdict
from operator import itemgetter
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict

####################################################################################################################
def initialize_observables_dict_from_tables(table_dir, analysis_config, parameterization, validation_indices):
    '''
    Initialize from .dat files into a dictionary of numpy arrays
      - We loop through all observables in the table directory for the given model and parameterization
      - We include only those observables:
         - That have sqrts,centrality specified in the analysis_config
         - Whose filename contains a string from analysis_config observable_list
      - We also separate out the design/predictions with indices in the validation set

    Note that all of the data points are the ratio of AA/pp

    :param str table_dir: directory where tables are located
    :param dict analysis_config: dictionary of analysis configuration
    :param str parameterization: name of qhat parameterization
    :param list validation_indices: list of design point indices to be used as validation set
    :return Return a dictionary with the following structure:
       observables['Data'][observable_label]['y'] -- value
                                            ['y_err'] -- total uncertainty (TODO: include uncertainty breakdowns)
                                            ['xmin'] -- bin lower edge (used only for plotting)
                                            ['xmax'] -- bin upper edge (used only for plotting)
       observables['Design'][parameterization] -- design points for a given parameterization
       observables['Prediction'][observable_label]['y'] -- value
                                                  ['y_err'] -- statistical uncertainty

       observables['Design_validation']... -- design points for validation set
       observables['Prediction_validation']... -- predictions for validation set

       where observable_label follows the convention from the table filenames:
           observable_label = f'{sqrts}__{system}__{observable_type}__{observable}__{subobservable}__{centrality}'
    :rtype dict
    '''
    print('Including the following observables:')

    # We will construct a dict containing all observables  
    observables = _recursive_defaultdict()

    #----------------------
    # Read experimental data 
    data_dir = os.path.join(table_dir, 'Data')
    for filename in os.listdir(data_dir):
        if _accept_observable(analysis_config, filename):

            data = np.loadtxt(os.path.join(data_dir, filename), ndmin=2)
            data_entry = {}
            data_entry['xmin'] = data[:,0]
            data_entry['xmax'] = data[:,1]
            data_entry['y'] = data[:,2]
            data_entry['y_err'] = data[:,3]

            observable_label, _ = _filename_to_labels(filename)
            observables['Data'][observable_label] = data_entry

            if 0 in data_entry['y']:
                sys.exit(f'{filename} has value=0')

    #----------------------
    # Read design points 
    design_dir = os.path.join(table_dir, 'Design')
    for filename in os.listdir(design_dir):

        if _filename_to_labels(filename)[1] == parameterization: 
            design_points = np.loadtxt(os.path.join(design_dir, filename), ndmin=2)

            # Separate training and validation sets into separate dicts
            with open(os.path.join(design_dir, filename)) as f:
                for line in f.readlines():
                    if 'Design point indices' in line:
                        indices = set([int(s) for s in line.split(':')[1].split()])
            training_indices_numpy = list(indices - set(validation_indices))
            validation_indices_numpy = list(indices.intersection(set(validation_indices)))
            observables['Design'] = design_points[training_indices_numpy]
            observables['Design_validation'] = design_points[validation_indices_numpy]

    #----------------------
    # Read predictions and uncertainty
    prediction_dir = os.path.join(table_dir, 'Prediction')
    for filename in os.listdir(prediction_dir):

        if 'values' in filename and parameterization in filename:
            if _accept_observable(analysis_config, filename):

                filename_prediction_values = filename
                filename_prediction_errors = filename.replace('values', 'errors') 
                observable_label, _ = _filename_to_labels(filename_prediction_values)

                prediction_values = np.loadtxt(os.path.join(prediction_dir, filename_prediction_values), ndmin=2)
                prediction_errors = np.loadtxt(os.path.join(prediction_dir, filename_prediction_errors), ndmin=2)

                # Separate training and validation sets into separate dicts
                with open(os.path.join(prediction_dir, filename_prediction_values)) as f:
                    for line in f.readlines():
                        if 'design_point' in line:
                            indices = set([int(s[12:]) for s in line.split('#')[1].split()])
                training_indices_numpy = list(indices - set(validation_indices))
                validation_indices_numpy = list(indices.intersection(set(validation_indices)))

                observables['Prediction'][observable_label]['y'] = np.take(prediction_values, training_indices_numpy, axis=1)
                observables['Prediction'][observable_label]['y_err'] = np.take(prediction_errors, training_indices_numpy, axis=1)

                observables['Prediction_validation'][observable_label]['y'] = np.take(prediction_values, validation_indices_numpy, axis=1)
                observables['Prediction_validation'][observable_label]['y_err'] = np.take(prediction_errors, validation_indices_numpy, axis=1)

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

####################################################################################################################
def write_dict_to_h5(results, output_dir, filename):
    '''
    Write nested dictionary of ndarray to hdf5 file
    Note: all keys should be strings

    :param dict results: (nested) dictionary to write
    :param str output_dir: directory to write to
    :param str filename: name of hdf5 file to create (will overwrite)
    '''
    print()

    print(f'Writing results to {output_dir}/{filename}...')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dicttoh5(results, os.path.join(output_dir, filename), overwrite_data=True)

    print('Done.')
    print()

####################################################################################################################
def read_dict_from_h5(input_dir, filename):
    '''
    Read dictionary of ndarrays from hdf5
    Note: all keys should be strings

    :param str input_dir: directory from which to read data
    :param str filename: name of hdf5 file to read
    '''
    print()
    print(f'Loading results from {input_dir}/{filename}...')

    results = h5todict(os.path.join(input_dir, filename))

    print('Done.')
    print()

    return results

####################################################################################################################
def predictions_matrix_from_h5(output_dir, filename):
    '''
    Initialize predictions from observables.h5 file into a single 2D array: 

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return 2darray Y: matrix of predictions at all design points (design_point_index, observable_bins) i.e. (n_samples, n_features)
    '''
    
    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename)

    # Sort observables, to keep well-defined ordering in matrix
    sorted_observable_list = sorted_observable_list_from_dict(observables)

    # Loop through sorted observables and concatenate them into a single 2D array: 
    #   (design_point_index, observable_bins) i.e. (n_samples, n_features)
    print(f'Doing PCA...')
    for i,observable_label in enumerate(sorted_observable_list):
        values = observables['Prediction'][observable_label]['y'].T
        if i==0:
            Y = values
        else:
            Y = np.concatenate([Y,values], axis=1)
    print(f'  Total shape of data (n_samples, n_features): {Y.shape}')

    return Y

####################################################################################################################
def design_array_from_h5(output_dir, filename):
    '''
    Initialize design array from observables.h5 file

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return 2darray design: array of design points
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename)
    design = observables['Design']
    return design

####################################################################################################################
def observable_label_to_keys(observable_label):
    '''
    Parse filename into individual keys

    :param str observable_label: observable label
    :return list of subobservables
    :rtype list
    '''

    observable_keys = observable_label.split('__')

    sqrts = observable_keys[0]
    system = observable_keys[1]
    observable_type = observable_keys[2]
    observable = observable_keys[3]
    subobserable = observable_keys[4]
    centrality = observable_keys[5]
    return sqrts, system, observable_type, observable, subobserable, centrality

####################################################################################################################
def sorted_observable_list_from_dict(observables):
    '''
    Define a sorted list of observable_labels from the keys of the observables dict, to keep well-defined ordering in matrix 

    :param dict observables: dictionary containing predictions/design/data
    :return list[str] sorted_observable_list: list of observable labels
    '''
    
    # Sort observables, to keep well-defined ordering in matrix
    sorted_observable_list = _sort_observable_labels(list(observables['Prediction'].keys()))
    return sorted_observable_list

#---------------------------------------------------------------
def _sort_observable_labels(unordered_observable_labels):
    '''
    Sort list of observable keys by observable_type, observable, subobservable, centrality, sqrts.
    TODO: Instead of a fixed sorting, we may want to allow the user to specify list of sort 
          criteria to apply, e.g. list of regex to iteratively sort by.

    :param list[str] observable_labels: unordered list of observable_label keys
    :return list[str] sorted_observable_labels: sorted observable_labels
    '''

    # First, sort the observable_labels to ensure an unambiguous ordering
    ordered_observable_labels = sorted(unordered_observable_labels)

    # Get the individual keys from the observable_label
    x = [observable_label_to_keys(observable_label) for observable_label in ordered_observable_labels]

    # Sort by (in order): observable_type, observable, subobservable, centrality, sqrts
    sorted_observable_label_tuples = sorted(x, key=itemgetter(2,3,4,5,0))

    # Reconstruct the observable_key 
    sorted_observable_labels = ['__'.join(x) for x in sorted_observable_label_tuples]

    return sorted_observable_labels

#---------------------------------------------------------------
def _filename_to_labels(filename):
    '''
    Parse filename to return observable_label, parameterization

    :param str filename: filename to parse
    :return list of subobservables and parameterization
    :rtype (list, str)
    '''

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
def _accept_observable(analysis_config, filename):
    '''
    Check if observable should be included in the analysis.
    It must:
      - Have sqrts,centrality specified in the analysis_config
      - Have a filename that contains a string from analysis_config observable_list

    :param dict analysis_config: dictionary of analysis configuration
    :param str filename: filename of table for the considered observable
    '''

    observable_label, _ = _filename_to_labels(filename)

    sqrts, _, _, _, _, centrality = observable_label_to_keys(observable_label)

    # Check sqrts
    if int(sqrts) not in analysis_config['sqrts_list']:
        return False

    # Check centrality
    accepted_centrality = False
    centrality_min, centrality_max = centrality.split('-')
    if int(centrality_min) >= analysis_config['centrality_range'][0]:
        if int(centrality_max) <= analysis_config['centrality_range'][1]:
            accepted_centrality = True
    if not accepted_centrality:
        return False

    # Check observable
    found_observable = False
    for observable_string in analysis_config['observable_list']:
        if observable_string in filename:
            found_observable = True
    if not found_observable:
        return False

    return True

#---------------------------------------------------------------
def _recursive_defaultdict():
    '''
    Create a nested defaultdict

    :return recursive defaultdict
    '''
    return defaultdict(_recursive_defaultdict)