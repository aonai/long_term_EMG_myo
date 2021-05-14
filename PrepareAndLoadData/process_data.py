import os, sys
import pickle
import numpy as np
import pandas as pd
from itertools import combinations


sys.path.insert(0,'../..')

pos = ['N', 'I', 'O', 'N', 'I', 'I', 'O', 'O', 'N', 'N',      # 1-10th day 
      'O', 'N', 'N', 'O', 'O', 'I', 'I', 'I', 'N', 'O',       # 11-20th day
      'O', 'I', 'O', 'I', 'I', 'N', 'N', 'I', 'N', 'O']       # 21-30th day

pos_label = [1, 2, 3, 1, 2, 2, 3, 3, 1, 1,
            3, 1, 1, 3, 3, 2, 2, 2, 1, 3,
            3, 2, 3, 2, 2, 1, 1, 2, 1, 3] # N: 1, I: 2, O: 3
# days correspond to N, I, and O positions
sessions_idx = [[],[],[]]
for idx, pl in enumerate(pos_label):
    sessions_idx[pl-1].append(idx+1)

day_num = 30
sub_num = 5
mov_num = 22
fs = 200
ch_num = 8
trial_num = 4

fs_pass = 15
fil_order = 5

win_size = 50            # 250ms window
win_inc = 10             # 50ms overlap

def getTSDfeatures_for_one_representation(vector):
    """
    Original by RamiKhushaba
        https://github.com/RamiKhushaba/getTSDfeat
    
    Edited by UlysseCoteAllard
		https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset

    % Implementation from Rami Khusaba
    % Time-domain power spectral moments (TD-PSD)
    % Using Fourier relations between time domina and frequency domain to
    % extract power spectral moments dircetly from time domain.
    %
    % Modifications
    % 17/11/2013  RK: Spectral moments first created.
    % 02/03/2014  AT: I added 1 to the function name to differentiate it from other versions from Rami
    % 01/02/2016  RK: Modifed this code intosomewhat deep structure
    %
    % References
    % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
    %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
    % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
    %     Neural Networks, vol. 55, pp. 42-58, 2014.
    """

    # Get the size of the input signal
    samples, channels = vector.shape

    if channels > samples:
        vector = np.transpose(vector)
        samples, channels = channels, samples

    # Root squared zero order moment normalized
    m0 = np.sqrt(np.nansum(vector ** 2, axis=0))[:, np.newaxis]
    m0 = m0 ** .1 / .1

    # Prepare derivatives for higher order moments
    d1 = np.diff(np.concatenate([np.zeros((1, channels)), vector], axis=0), n=1, axis=0)
    d2 = np.diff(np.concatenate([np.zeros((1, channels)), d1], axis=0), n=1, axis=0)

    # Root squared 2nd and 4th order moments normalized
    m2 = (np.sqrt(np.nansum(d1 ** 2, axis=0)) / (samples - 1))[:, np.newaxis]
    m2 = m2 ** .1 / .1

    m4 = (np.sqrt(np.nansum(d2 ** 2, axis=0)) / (samples - 1))[:, np.newaxis]
    m4 = m4 ** .1 / .1

    # Sparseness
    sparsi = m0 / np.sqrt(np.abs(np.multiply((m0 - m2) ** 2, (m0 - m4) ** 2)))

    # Irregularity Factor
    IRF = m2 / np.sqrt(np.multiply(m0, m4))

    # Coefficient of Variation
    tmp = np.nanmean(vector, axis=0)
    if 0 in tmp:   # avoid divison by zero case 
        tmp[tmp==0] = 1e-10
    COV = (np.nanstd(vector, axis=0, ddof=1) / tmp)[:, np.newaxis]

    # Teager-Kaiser energy operator
    TEA = np.nansum(d1 ** 2 - np.multiply(vector[0:samples, :], d2), axis=0)[:, np.newaxis]

    # All features together
    STDD = np.nanstd(m0, axis=0, ddof=1)[:, np.newaxis]

    if channels > 2:
        Feat = np.concatenate((m0 / STDD, (m0 - m2) / STDD, (m0 - m4) / STDD, sparsi, IRF, COV, TEA), axis=0)
    else:
        Feat = np.concatenate((m0, m0 - m2, m0 - m4, sparsi, IRF, COV, TEA), axis=0)

    Feat = np.log(np.abs(Feat)).flatten()

    return Feat

def getTSD(all_channels_data_in_window):
    """
    Original by RamiKhushaba
        https://github.com/RamiKhushaba/getTSDfeat
    
    Edited by UlysseCoteAllard
		https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset

    Note: this uses window emg directly instead of extracting a window then apply TSD like in Rami's code
    
    %Implementation from Rami Khusaba adapted to our code
    % References
    % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
    %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
    % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
    %     Neural Networks, vol. 55, pp. 42-58, 2014.
    """
    # x should be a numpy array
    all_channels_data_in_window = np.swapaxes(np.array(all_channels_data_in_window), 1, 0)

    if len(all_channels_data_in_window.shape) == 1:
        all_channels_data_in_window = all_channels_data_in_window[:, np.newaxis]

    datasize = all_channels_data_in_window.shape[0]
    Nsignals = all_channels_data_in_window.shape[1]

    # prepare indices of each 2 channels combinations
    # NCC = Number of channels to combine
    NCC = 2
    Indx = np.array(list(combinations(range(Nsignals), NCC)))   # (28,2)

    # allocate memory
    # define the number of features per channel
    NFPC = 7

    # Preallocate memory
    feat = np.zeros((Indx.shape[0] * NFPC + Nsignals * NFPC))

    # Step1.1: Extract between-channels features
    ebp = getTSDfeatures_for_one_representation(
        all_channels_data_in_window[:, Indx[:, 0]] - all_channels_data_in_window[:, Indx[:, 1]])
    efp = getTSDfeatures_for_one_representation(
        np.log(
            (all_channels_data_in_window[:, Indx[:, 0]] - all_channels_data_in_window[:, Indx[:, 1]]) ** 2 + np.spacing(
                1)) ** 2)
    # Step 1.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[range(Indx.shape[0] * NFPC)] = num / den

    # Step2.1: Extract within-channels features
    ebp = getTSDfeatures_for_one_representation(all_channels_data_in_window)
    efp = getTSDfeatures_for_one_representation(np.log((all_channels_data_in_window) ** 2 + np.spacing(1)) ** 2)
    # Step2.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[np.max(range(Indx.shape[0] * NFPC)) + 1:] = num / den
    return feat


def format_examples(emg_examples, window_size=50, size_non_overlap=10):
    """ 
    emg_examples: list of emg signals, each row represent one recording of a 8 channel emg
    feature_set_function
    window_size: analysis window size
    size_non_overlap: length of non-overlap portion between each analysis window

    Returns:
        formated_examples: (252,) array 
                including 7 features for each channel and for each two combination of channel signals
    """
    formated_examples = []
    example = []
    for emg_vector in emg_examples:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))
        
        # store one window_size of signal
        if len(example) >= window_size:
            if not np.sum(example) == 0:   # avoid all zero signals
                featured_example = getTSD(example.transpose())
                formated_examples.append(np.array(featured_example).transpose().flatten())
            else:
                formated_examples.append(np.zeros((252)))
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]
    return formated_examples

def read_files_to_format_training_session(path_folder_examples, day_num,
                                          number_of_cycles, number_of_gestures, window_size,
                                          size_non_overlap):
    """
    path_folder_examples: path to load training data
    feature_set_function
    number_of_cycles: number of trials recorded for each motion
    number_of_gestures
    window_size: analysis window size
    size_non_overlap: length of non-overlap portion between each analysis window
    
    shape(formated_example) = (26, 50, 8)
    """
    examples_training, labels_training = [], []
    
    for cycle in range(1, number_of_cycles+1):
        examples, labels = [], []
        for gesture_index in range(1, number_of_gestures+1):
            read_file = path_folder_examples + "/D" + str(day_num) + "M" + str(gesture_index) + "T" + str(cycle) + ".csv"
            # print("      READ ", read_file)
            examples_to_format = pd.read_csv(read_file, header=None).to_numpy()
            # each file contains 15s (300 rows) of 8 channel signals 
            # print("            data = ", np.shape(examples_to_format))
            
            examples_formatted = format_examples(examples_to_format,
                                     window_size=window_size,
                                     size_non_overlap=size_non_overlap)
            # print("            formated = ", np.shape(examples_formatted))

            examples.extend(examples_formatted)
            labels.extend(np.ones(len(examples_formatted)) * (gesture_index-1))
            
        # print("   SHAPE SESSION ", cycle, " EXAMPLES: ", np.shape(examples))
        examples_training.append(examples)
        labels_training.append(labels)
        # print("   SHAPE ALL SESSION EXAMPLES: ", np.shape(examples_training))  

    return examples_training, labels_training

def get_data_and_process_it_from_file(path, number_of_gestures=22, number_of_cycles=4, window_size=50, size_non_overlap=10):

    """
    Args:
        path: path to load training data
        number_of_gestures
        number_of_cycles: number of trials recorded for each motion
        window_size: analysis window size
        size_non_overlap: length of non-overlap portion between each analysis window

    Returns:
        loaded data dictionary containing `examples_training` and `labels_training`
    """
    examples_training_sessions_datasets, labels_training_sessions_datasets = [], []

    # load one participant for now
    for index_participant in range(1,4):
        # load one participant data 
        folder_participant = "sub" + str(index_participant)
        examples_participant_training_sessions, labels_participant_training_sessions = [], []
        for days_of_current_session in sessions_idx:
            print("process data in days ", days_of_current_session)
            examples_per_session, labels_per_session = [], []
            for day_num in days_of_current_session:
                path_folder_examples = path + "/" + folder_participant + "/day" + str(day_num)
                # print("current dr = ", day_num)
                
                examples_training, labels_training  = \
                    read_files_to_format_training_session(path_folder_examples=path_folder_examples,
                                                        day_num = day_num,
                                                        number_of_cycles=number_of_cycles,
                                                        number_of_gestures=number_of_gestures,
                                                        window_size=window_size,
                                                        size_non_overlap=size_non_overlap)
                examples_per_session.extend(examples_training)
                labels_per_session.extend(labels_training)
            examples_participant_training_sessions.append(examples_per_session)
            labels_participant_training_sessions.append(labels_per_session)
            print("@ traning sessions = ", np.shape(examples_participant_training_sessions))


        # participants_num x sessions_num(3) x days_per_session(10)*trail_per_day(4) x #examples_window*#mov(26*22=572) x window_size x channel_num
        print('traning examples ', np.shape(examples_participant_training_sessions))
        examples_training_sessions_datasets.append(examples_participant_training_sessions)
        print('all traning examples ', np.shape(examples_training_sessions_datasets))

        # participants_num x sessions_num(3) x days_per_session(10)*trail_per_day(4) x #examples_window*#mov(26*22=572)
        print('traning labels ', np.shape(labels_participant_training_sessions))
        labels_training_sessions_datasets.append(labels_participant_training_sessions)
        print('all traning labels ', np.shape(labels_training_sessions_datasets))
    
    # store processed data to dictionary
    dataset_dictionnary = {"examples_training": np.array(examples_training_sessions_datasets, dtype=object),
                        "labels_training": np.array(labels_training_sessions_datasets, dtype=object)}
    return dataset_dictionnary


def read_data_training(path, store_path, number_of_gestures=22, number_of_cycles=4, window_size=50, size_non_overlap=10):
    """
    path: path to load training data
    store_path: path to stored loaded data dictionary
        contains `examples_training` and `labels_training`
    number_of_gestures
    number_of_cycles: number of trials recorded for each motion
    window_size: analysis window size
    size_non_overlap: length of non-overlap portion between each analysis window
    """
    print("Loading and preparing Training datasets...")
    dataset_dictionnary = get_data_and_process_it_from_file(path=path, number_of_gestures=number_of_gestures,
                                                            number_of_cycles=number_of_cycles, window_size=window_size,
                                                            size_non_overlap=size_non_overlap)

    # store dictionary to pickle
    training_session_dataset_dictionnary = {}
    training_session_dataset_dictionnary["examples_training"] = dataset_dictionnary["examples_training"]
    training_session_dataset_dictionnary["labels_training"] = dataset_dictionnary["labels_training"]

    with open(store_path + "/training_session.pickle", 'wb') as f:
        pickle.dump(training_session_dataset_dictionnary, f, pickle.HIGHEST_PROTOCOL)


