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
import logging
from collections import defaultdict
from operator import itemgetter
import numpy as np
from silx.io.dictdump import dicttoh5, h5todict


logger = logging.getLogger(__name__)


####################################################################################################################
def initialize_observables_dict_from_tables(table_dir, analysis_config, parameterization):
    '''
    Initialize from .dat files into a dictionary of numpy arrays
      - We loop through all observables in the table directory for the given model and parameterization
      - We include only those observables:
         - That have sqrts, centrality specified in the analysis_config
         - Whose filename contains a string from analysis_config observable_list
      - We also separate out the design/predictions with indices in the validation set

    Note that all of the data points are the ratio of AA/pp

    :param str table_dir: directory where tables are located
    :param dict analysis_config: dictionary of analysis configuration
    :param str parameterization: name of qhat parameterization
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
    logger.info('Including the following observables:')

    # We will construct a dict containing all observables
    observables = _recursive_defaultdict()

    # We separate out the validation indices specified in the config
    validation_range = analysis_config['validation_indices']
    validation_indices = range(validation_range[0], validation_range[1])

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
                msg = f'{filename} has value=0'
                raise ValueError(msg)

    #----------------------
    # Read design points
    design_dir = os.path.join(table_dir, 'Design')
    for filename in os.listdir(design_dir):

        if _filename_to_labels(filename)[1] == parameterization:
            design_points = np.loadtxt(os.path.join(design_dir, filename), ndmin=2)

            # Separate training and validation sets into separate dicts
            training_indices_numpy, validation_indices_numpy = _split_training_validation_indices(validation_indices, table_dir, parameterization)
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
                    logger.info(f'WARNING: {filename_prediction_values} has value=0 at design points {np.where(prediction_values == 0)[1]}')

                # Check that data and prediction have same observables with the same size
                if observable_label not in observables['Data']:
                    data_keys = observables['Data'].keys()
                    msg = f'{observable_label} not found in observables[Data]: {data_keys}'
                    raise ValueError(msg)

                data_size = observables['Data'][observable_label]['y'].shape[0]
                prediction_size = observables['Prediction'][observable_label]['y'].shape[0]
                if data_size != prediction_size:
                    msg = f'({filename_prediction_values}) has different shape ({prediction_size}) than Data ({data_size}).'
                    raise ValueError(msg)

    #----------------------
    # Construct covariance matrices

    #----------------------
    # Print observables that we will use
    [logger.info(f'  {s}') for s in sorted(observables['Prediction'].keys())]

    return observables

####################################################################################################################
def write_dict_to_h5(results, output_dir, filename, verbose=True):
    '''
    Write nested dictionary of ndarray to hdf5 file
    Note: all keys should be strings

    :param dict results: (nested) dictionary to write
    :param str output_dir: directory to write to
    :param str filename: name of hdf5 file to create (will overwrite)
    '''
    if verbose:
        logger.info("")
        logger.info(f'Writing results to {output_dir}/{filename}...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dicttoh5(results, os.path.join(output_dir, filename), update_mode="modify")

    if verbose:
        logger.info('Done.')
        logger.info("")

####################################################################################################################
def read_dict_from_h5(input_dir, filename, verbose=True):
    '''
    Read dictionary of ndarrays from hdf5
    Note: all keys should be strings

    :param str input_dir: directory from which to read data
    :param str filename: name of hdf5 file to read
    '''
    if verbose:
        logger.info("")
        logger.info(f'Loading results from {input_dir}/{filename}...')

    results = h5todict(os.path.join(input_dir, filename))

    if verbose:
        logger.info('Done.')
        logger.info("")

    return results

####################################################################################################################
def predictions_matrix_from_h5(output_dir, filename, validation_set=False):
    '''
    Initialize predictions from observables.h5 file into a single 2D array:

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return 2darray Y: matrix of predictions at all design points (design_point_index, observable_bins) i.e. (n_samples, n_features)
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename, verbose=False)

    # Sort observables, to keep well-defined ordering in matrix
    sorted_observable_list = sorted_observable_list_from_dict(observables)

    # Set dictionary key
    if validation_set:
        prediction_label = 'Prediction_validation'
    else:
        prediction_label = 'Prediction'

    # Loop through sorted observables and concatenate them into a single 2D array:
    #   (design_point_index, observable_bins) i.e. (n_samples, n_features)
    for i,observable_label in enumerate(sorted_observable_list):
        values = observables[prediction_label][observable_label]['y'].T
        if i==0:
            Y = values
        else:
            Y = np.concatenate([Y,values], axis=1)
    logger.info(f'  Total shape of {prediction_label} data (n_samples, n_features): {Y.shape}')

    return Y

####################################################################################################################
def design_array_from_h5(output_dir, filename, validation_set=False):
    '''
    Initialize design array from observables.h5 file

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return 2darray design: array of design points
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename, verbose=False)
    if validation_set:
        design = observables['Design_validation']
    else:
        design = observables['Design']
    return design

####################################################################################################################
def data_array_from_h5(output_dir, filename, observable_table_dir=None):
    '''
    Initialize data array from observables.h5 file

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return 2darray data: arrays of data points (columns of data[observable_label]: xmin xmax y y_err)
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename, verbose=False)
    data = observables['Data']

    # Check that data matches original table (if observable_table_dir is specified)
    if observable_table_dir:
        data_table_dir = os.path.join(observable_table_dir, 'Data')
        for observable_label in observables['Data'].keys():
            data_table_filename = f'Data__{observable_label}.dat'
            data_table = np.loadtxt(os.path.join(data_table_dir, data_table_filename), ndmin=2)
            assert np.allclose(data[observable_label]['xmin'], data_table[:,0])
            assert np.allclose(data[observable_label]['xmax'], data_table[:,1])
            assert np.allclose(data[observable_label]['y'], data_table[:,2])
            assert np.allclose(data[observable_label]['y_err'] , data_table[:,3])

    return data

####################################################################################################################
def prediction_dict_from_matrix(Y, observables, config=None, validation_set=False):
    '''
    Translate matrix of stacked observables to a dict of matrices per observable

    :param ndarray Y: 2D array: (design_point_index, observable_bins)
    :param dict observables: dict
    :param str observable_table_dir: (optional, only needed to check against table values)
    :param str parameterization: (optional, only needed to check against table values)
    :param str validation_range: (optional, only needed to check against table values)
    :param str validation_set: (optional, only needed to check against table values)
    :return dict[ndarray] Y_dict: dict with ndarray for each observable
    '''

    Y_dict = {}

    # Loop through sorted list of observables
    sorted_observable_list = sorted_observable_list_from_dict(observables)
    current_bin = 0
    if validation_set:
        prediction_key = 'Prediction_validation'
    else:
        prediction_key = 'Prediction'
    for observable_label in sorted_observable_list:
        n_bins = observables[prediction_key][observable_label]['y'].shape[0]
        Y_dict[observable_label] = Y[:,current_bin:current_bin+n_bins]
        current_bin += n_bins

    # Check that the total number of bins is correct
    assert current_bin == Y.shape[1]

    # Check that prediction matches original table (if observable_table_dir, parameterization,validation_indices are specified)
    # If validation_set, select the validation indices; otherwise, select the training indices
    if config:

        validation_range = config.analysis_config['validation_indices']
        validation_indices = range(validation_range[0], validation_range[1])
        training_indices_numpy, validation_indices = _split_training_validation_indices(validation_indices, config.observable_table_dir, config.parameterization)
        if validation_set:
            indices_numpy = validation_indices
        else:
            indices_numpy = training_indices_numpy

        prediction_table_dir = os.path.join(config.observable_table_dir, 'Prediction')
        for observable_label in sorted_observable_list:
            prediction_table_filename = f'Prediction__{config.parameterization}__{observable_label}__values.dat'
            prediction_table = np.loadtxt(os.path.join(prediction_table_dir, prediction_table_filename), ndmin=2)
            prediction_table_selected = np.take(prediction_table, indices_numpy, axis=1).T
            assert np.allclose(Y_dict[observable_label], prediction_table_selected), \
                               f'{observable_label} (design point 0) \n prediction: {Y_dict[observable_label][0,:]} \n prediction (table): {prediction_table_selected[0,:]}'

    return Y_dict

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
    centrality_min, centrality_max = centrality.split('-')
    # Validation
    # Provided a single centrality range - convert to a list of ranges
    centrality_ranges = analysis_config['centrality_range']
    if not isinstance(centrality_ranges[0], list):
        centrality_ranges = [list(centrality_ranges)]

    accepted_centrality = False
    for (selected_cent_min, selected_cent_max) in centrality_ranges:
        if int(centrality_min) >= selected_cent_min:
            if int(centrality_max) <= selected_cent_max:
                accepted_centrality = True
                # Bail out - no need to keep looping if it's already accepted
                break
    if not accepted_centrality:
        return False

    # Check observable
    # Select observables based on the input list, with the possibility of excluding some
    # observables with additional selection strings (eg. remove one experiment from the
    # observables for an exploratory analysis).
    config_observable_list = analysis_config['observable_list']
    config_observable_exclude_list = analysis_config.get("observable_exclude_list", [])
    observable_in_include_list = any([observable_string in filename for observable_string in config_observable_list])
    observable_in_exclude_list = any([exclude in filename for exclude in config_observable_exclude_list])

    found_observable = (observable_in_include_list and not observable_in_exclude_list)

    # Helpful for cross checking when debugging
    if observable_in_exclude_list:
        logger.debug(f"Excluding observable '{filename}' due to exclude list. {found_observable=}")

    return found_observable

#---------------------------------------------------------------
def _split_training_validation_indices(validation_indices, observable_table_dir, parameterization):
    '''
    Get numpy indices of training and validation sets

    :param list[int] validation_indices: list of validation indices
    :param str observable_table_dir: location of table dir
    :param str parameterization: qhat parameterization type
    '''

    # Get training set or validation set
    design_table_dir = os.path.join(observable_table_dir, 'Design')
    design_filename = f'Design__{parameterization}.dat'
    with open(os.path.join(design_table_dir, design_filename)) as f:
        for line in f.readlines():
            if 'Design point indices' in line:
                indices = set([int(s) for s in line.split(':')[1].split()])
    training_indices_numpy = list(indices - set(validation_indices))
    validation_indices_numpy = list(indices.intersection(set(validation_indices)))

    return training_indices_numpy, validation_indices_numpy

#---------------------------------------------------------------
def _recursive_defaultdict():
    '''
    Create a nested defaultdict

    :return recursive defaultdict
    '''
    return defaultdict(_recursive_defaultdict)