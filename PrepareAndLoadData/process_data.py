import sys
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

from PrepareAndLoadData.calculate_spectrograms import calculate_single_example

sys.path.insert(0,'../..')


"""  Params for Longterm_myo dataset
Dataset recorded by https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset

N = neutral position
I = inward rotation
O = outward rotation

"""
pos = ['N', 'I', 'O', 'N', 'I', 'I', 'O', 'O', 'N', 'N',      # 1-10th day 
      'O', 'N', 'N', 'O', 'O', 'I', 'I', 'I', 'N', 'O',       # 11-20th day
      'O', 'I', 'O', 'I', 'I', 'N', 'N', 'I', 'N', 'O']       # 21-30th day

pos_label = [1, 2, 3, 1, 2, 2, 3, 3, 1, 1,
            3, 1, 1, 3, 3, 2, 2, 2, 1, 3,
            3, 2, 3, 2, 2, 1, 1, 2, 1, 3] # N: 1, I: 2, O: 3
# record days correspond to N, I, and O positions
index_of_sessions = [[],[],[]]
for idx, pl in enumerate(pos_label):
    index_of_sessions[pl-1].append(idx+1)

day_num = 30            # total number of days recorded
sub_num = 5             # total number of subjects recorded
mov_num = 22            # total number of gestures recorded
fs = 200                # myo sampling frequency 
ch_num = 8              # myo channel number
trial_num = 4           # total number of trials recorded per gesture  
# analysis window for sEMG
win_size = 50            # 250ms window 
win_inc = 10             # 50ms overlap


def getTSDfeatures_for_one_representation(vector):
    """
    Original by RamiKhushaba
        https://github.com/RamiKhushaba/getTSDfeat
    
    Edited by UlysseCoteAllard
		https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset

    Refer to Equations in [3]

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

    [3] R. N. Khushaba, A. H. Al-Timemy, A. Al-Ani and A. Al-Jumaily, "A Framework of Temporal-Spatial Descriptors-Based 
    Feature Extraction for Improved Myoelectric Pattern Recognition," in IEEE Transactions on Neural Systems and
    Rehabilitation Engineering, vol. 25, no. 10, pp. 1821-1831, Oct. 2017, doi: 10.1109/TNSRE.2017.2687520.
    """

    # Get the size of the input signal
    samples, channels = vector.shape

    if channels > samples:
        vector = np.transpose(vector)
        samples, channels = channels, samples

    # Root squared zero order moment normalized  
    # Equation (2) and (6) with lambda = 0.1
    m0 = np.sqrt(np.nansum(vector ** 2, axis=0))[:, np.newaxis]
    m0 = m0 ** .1 / .1

    # Prepare derivatives for higher order moments
    # Euqation (3) and (6)      
    d1 = np.diff(np.concatenate([np.zeros((1, channels)), vector], axis=0), n=1, axis=0)
    d2 = np.diff(np.concatenate([np.zeros((1, channels)), d1], axis=0), n=1, axis=0)

    # Root squared 2nd and 4th order moments normalized
    # Equation (4) and (6)
    m2 = (np.sqrt(np.nansum(d1 ** 2, axis=0)) / (samples - 1))[:, np.newaxis]
    m2 = m2 ** .1 / .1

    # Equation (5) and (6)
    m4 = (np.sqrt(np.nansum(d2 ** 2, axis=0)) / (samples - 1))[:, np.newaxis]
    m4 = m4 ** .1 / .1

    # Sparseness
    # Euqation (8)
    sparsi = m0 / np.sqrt(np.abs(np.multiply((m0 - m2) ** 2, (m0 - m4) ** 2)))

    # Irregularity Factor
    # Equation (9)
    IRF = m2 / np.sqrt(np.multiply(m0, m4))

    # Coefficient of Variation
    # Equation (10)
    tmp = np.nanmean(vector, axis=0)
    if 0 in tmp:   # avoid divison by zero case 
        tmp[tmp==0] = 1e-10
    COV = (np.nanstd(vector, axis=0, ddof=1) / tmp)[:, np.newaxis]

    # Teager-Kaiser energy operator
    # Equation (11)
    TEA = np.nansum(d1 ** 2 - np.multiply(vector[0:samples, :], d2), axis=0)[:, np.newaxis]

    # All features together
    # Maybe similar to Equation (11)
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

    Note: this uses window emg directly instead of extracting a window then apply TSD like in Rami's code. Calculation
    is mainly based on equations explained in [3].

    Note2: it looks like the window size needs to be at least 28 to have this functions working; otherwise, the calcualted 
            features will not be able to fill up all the combinations 
    
    %Implementation from Rami Khusaba adapted to our code
    % References
    % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
    %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
    % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
    %     Neural Networks, vol. 55, pp. 42-58, 2014.

    [3] R. N. Khushaba, A. H. Al-Timemy, A. Al-Ani and A. Al-Jumaily, "A Framework of Temporal-Spatial Descriptors-Based 
        Feature Extraction for Improved Myoelectric Pattern Recognition," in IEEE Transactions on Neural Systems and
        Rehabilitation Engineering, vol. 25, no. 10, pp. 1821-1831, Oct. 2017, doi: 10.1109/TNSRE.2017.2687520.
    """
    # x should be a numpy array
    all_channels_data_in_window = np.swapaxes(np.array(all_channels_data_in_window), 1, 0)

    if len(all_channels_data_in_window.shape) == 1:
        all_channels_data_in_window = all_channels_data_in_window[:, np.newaxis]

    datasize = all_channels_data_in_window.shape[0]
    Nsignals = all_channels_data_in_window.shape[1]
    # print("all_channels_data_in_window  = ", all_channels_data_in_window.shape)

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
    # print("num = ", np.shape(num))
    # print("put into feat = ", np.shape(range(Indx.shape[0] * NFPC)))
    feat[range(Indx.shape[0] * NFPC)] = num / den

    # Step2.1: Extract within-channels features
    ebp = getTSDfeatures_for_one_representation(all_channels_data_in_window)
    efp = getTSDfeatures_for_one_representation(np.log((all_channels_data_in_window) ** 2 + np.spacing(1)) ** 2)
    # Step2.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[np.max(range(Indx.shape[0] * NFPC)) + 1:] = num / den
    return feat


def format_examples(emg_examples, window_size=50, size_non_overlap=10, spectrogram=False):
    """ 
    Process EMG signals and then put into one window

    Args:
        emg_examples: list of emg signals, each row represent one recording of a 8 channel emg
        window_size: analysis window size
        size_non_overlap: length of non-overlap portion between each analysis window
        spectrogram: whether to process sEMG into spectrograms 

    Returns:
        formated_examples: (252,) array including 7 features for each channel and for each two 
                            combination of channel signals
                            or spectrogram array of shape (4, 8, 10) 
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
            if spectrogram: # get spectrogram
                spectrogram_example = calculate_single_example(example.transpose(), frequency=fs)
                formated_examples.append(np.array(spectrogram_example))
            else: # get TSD features 
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
                                          size_non_overlap,include_gestures=None, spectrogram=False):
    """
    Read csv files in one day, put raw signals into an array, then process the signals using TSD function 
    and put into windows 

    Args:
        path_folder_examples: path to load raw signals
        day_num: which day to load data from, should be integer between 1~30
        number_of_cycles: number of trials recorded for each motion
        number_of_gestures: numer of gestures recorded 
        window_size: analysis window size
        size_non_overlap: length of non-overlap portion between each analysis window
        include_gestures: list of gestures (in integer) to include in processed signals; include all gestures if None
        spectrogram: whether to process sEMG into spectrograms 

    Returns:
        examples_training, labels_training: ndarray of processed signal windows and corresponding labels
    """
    examples_training, labels_training = [], []
    
    for cycle in range(1, number_of_cycles+1):
        examples, labels = [], []
        if include_gestures:
            for num_idx, gesture_index in enumerate(include_gestures):
                read_file = path_folder_examples + "/D" + str(day_num) + "M" + str(gesture_index) + "T" + str(cycle) + ".csv"
                # print("      READ ", read_file)
                examples_to_format = pd.read_csv(read_file, header=None).to_numpy()
                # each file contains 15s (300 rows) of 8 channel signals 
                # print("            data = ", np.shape(examples_to_format))
                
                examples_formatted = format_examples(examples_to_format,
                                        window_size=window_size,
                                        size_non_overlap=size_non_overlap,
                                        spectrogram=spectrogram)
                # print("            formated = ", np.shape(examples_formatted))

                examples.extend(examples_formatted)
                labels.extend(np.ones(len(examples_formatted)) * num_idx)
        else:
            for gesture_index in range(1, number_of_gestures+1):
                read_file = path_folder_examples + "/D" + str(day_num) + "M" + str(gesture_index) + "T" + str(cycle) + ".csv"
                # print("      READ ", read_file)
                examples_to_format = pd.read_csv(read_file, header=None).to_numpy()
                # each file contains 15s (300 rows) of 8 channel signals 
                # print("            data = ", np.shape(examples_to_format))
                
                examples_formatted = format_examples(examples_to_format,
                                        window_size=window_size,
                                        size_non_overlap=size_non_overlap,
                                        spectrogram=spectrogram)
                # print("            formated = ", np.shape(examples_formatted))

                examples.extend(examples_formatted)
                labels.extend(np.ones(len(examples_formatted)) * (gesture_index-1))
            
        # print("   SHAPE SESSION ", cycle, " EXAMPLES: ", np.shape(examples))
        examples_training.append(examples)
        labels_training.append(labels)
        # print("   SHAPE ALL SESSION EXAMPLES: ", np.shape(examples_training))  

    return examples_training, labels_training

def get_data_and_process_it_from_file(path, number_of_gestures=22, number_of_cycles=4, window_size=50, 
                                        size_non_overlap=10, num_participant=5, sessions_to_include = [0,1,2], 
                                        switch=0, start_at_participant=1, include_gestures=None, spectrogram=False):
    
    """
    Wrapper for loading data from desired folders. 

    Args:
        path: path to load training data
        number_of_gestures: numer of gestures recorded 
        number_of_cycles: number of trials recorded for each motion
        window_size: analysis window size
        size_non_overlap: length of non-overlap portion between each analysis window
        num_participant: numer of participants to include; should be integer between 1~5
        sessions_to_include: array of integers defining which session to include in across day training
        swtich: determine which of the following case to run
                case 0 = across wearing locations; processed dataset will be in form participants_num x sessions_num(3) 
                case 1 = across subjects; processed dataset will be in form sessions_num(3) x participants_num;
                        when choosing case 1, remember to sepcify which subject will be used for base model 
                        in start_at_participant
                case 2 = across days (among the same subject and wearing location);
                    when choosing case 2, remember to sepcify which session to include in sessions_to_include;
                    one assumption for this case is that only one session (wearing location) is included for training
        start_at_participant: indicates which subject is the used as base model for DANN and SCADANN training 
                                in across subject training 
        include_gestures: list of gestures (in integer) to include in processed signals; include all gestures if None
        spectrogram: whether to process sEMG into spectrograms 

    Returns:
        data dictionary containing an array of `examples_training` and `labels_training`
    """
    examples_training_sessions_datasets, labels_training_sessions_datasets = [], []

    # load one participant for now
    if switch == 1:
        for session_idx in sessions_to_include:
            # load one participant data 
            examples_participant_training_sessions, labels_participant_training_sessions = [], []
            for index_participant in range(start_at_participant,start_at_participant+num_participant):
                folder_participant = "sub" + str(index_participant)
                days_of_current_session = index_of_sessions[session_idx]
                if session_idx in sessions_to_include:
                    print("session ", session_idx, " --- process data in days ", days_of_current_session)
                    examples_per_session, labels_per_session = [], []
                    for day_num in days_of_current_session:
                        path_folder_examples = path + "/" + folder_participant + "/day" + str(day_num)
                        # print("current dr = ", day_num)
                        print("READ ", "Sub", index_participant, "_Loc", sessions_to_include[0], "_Day", day_num)
                        
                        examples_training, labels_training  = \
                            read_files_to_format_training_session(path_folder_examples=path_folder_examples,
                                                                day_num = day_num,
                                                                number_of_cycles=number_of_cycles,
                                                                number_of_gestures=number_of_gestures,
                                                                window_size=window_size,
                                                                size_non_overlap=size_non_overlap,
                                                                include_gestures=include_gestures,
                                                                spectrogram=spectrogram)
                        examples_per_session.extend(examples_training)
                        labels_per_session.extend(labels_training)
                    examples_participant_training_sessions.append(examples_per_session)
                    labels_participant_training_sessions.append(labels_per_session)
                    print("@ traning sessions = ", np.shape(examples_participant_training_sessions))

            print('traning examples ', np.shape(examples_participant_training_sessions))
            print('traning labels ', np.shape(labels_participant_training_sessions))  
                    
            examples_training_sessions_datasets.append(examples_participant_training_sessions)
            labels_training_sessions_datasets.append(labels_participant_training_sessions)

            # sessions_num(3) x participants_num x days_per_session(10)*trail_per_day(4) x #examples_window*#mov(26*22=572) x window_size x channel_num
            # sessions_num(3) x participants_num x days_per_session(10)*trail_per_day(4) x #examples_window*#mov(26*22=572)
            print('all traning examples ', np.shape(examples_training_sessions_datasets))
            print('all traning labels ', np.shape(labels_training_sessions_datasets))

    elif switch == 0:
        for index_participant in range(1, 1+num_participant):
            # load one participant data 
            folder_participant = "sub" + str(index_participant)
            examples_participant_training_sessions, labels_participant_training_sessions = [], []
            for session_idx in sessions_to_include:
                days_of_current_session = index_of_sessions[session_idx]
                if session_idx in sessions_to_include:
                    print("session ", session_idx, " --- process data in days ", days_of_current_session)
                    examples_per_session, labels_per_session = [], []
                    for day_num in days_of_current_session:
                        path_folder_examples = path + "/" + folder_participant + "/day" + str(day_num)
                        # print("current dr = ", day_num)
                        print("READ ", "Sub", index_participant, "_Loc", sessions_to_include[0], "_Day", day_num)
                        examples_training, labels_training  = \
                            read_files_to_format_training_session(path_folder_examples=path_folder_examples,
                                                                day_num = day_num,
                                                                number_of_cycles=number_of_cycles,
                                                                number_of_gestures=number_of_gestures,
                                                                window_size=window_size,
                                                                size_non_overlap=size_non_overlap,
                                                                include_gestures=include_gestures,
                                                                spectrogram=spectrogram)
                        examples_per_session.extend(examples_training)
                        labels_per_session.extend(labels_training)
                    examples_participant_training_sessions.append(examples_per_session)
                    labels_participant_training_sessions.append(labels_per_session)
                    print("@ traning sessions = ", np.shape(examples_participant_training_sessions))

            print('traning examples ', np.shape(examples_participant_training_sessions))
            print('traning labels ', np.shape(labels_participant_training_sessions))  
                    
            examples_training_sessions_datasets.append(examples_participant_training_sessions)
            labels_training_sessions_datasets.append(labels_participant_training_sessions)
            
            # participants_num x sessions_num(3) x days_per_session(10)*trail_per_day(4*num_participant) x #examples_window*#mov(26*22=572) x window_size x channel_num
            # participants_num x sessions_num(3) x days_per_session(10)*trail_per_day(4*num_participant) x #examples_window*#mov(26*22=572)
            print('all traning examples ', np.shape(examples_training_sessions_datasets))
            print('all traning labels ', np.shape(labels_training_sessions_datasets))

    elif switch == 2:
        days_of_current_session = index_of_sessions[sessions_to_include[0]]
        examples_participant_training_sessions, labels_participant_training_sessions = [], []
        print("session ", sessions_to_include[0], " --- process data in days ", days_of_current_session)
        for index_participant in range(1,1+num_participant):
            examples_per_session, labels_per_session = [], []
            for day_num in days_of_current_session:
                # load one participant data 
                folder_participant = "sub" + str(index_participant)
        
                path_folder_examples = path + "/" + folder_participant + "/day" + str(day_num)
                # print("current dr = ", day_num)
                print("READ ", "Sub", index_participant, "_Loc", sessions_to_include[0], "_Day", day_num)
                examples_training, labels_training  = \
                    read_files_to_format_training_session(path_folder_examples=path_folder_examples,
                                                        day_num = day_num,
                                                        number_of_cycles=number_of_cycles,
                                                        number_of_gestures=number_of_gestures,
                                                        window_size=window_size,
                                                        size_non_overlap=size_non_overlap,
                                                        include_gestures=include_gestures,
                                                        spectrogram=spectrogram)
                examples_per_session.append(examples_training)
                labels_per_session.append(labels_training)
                print("examples_per_session = ", np.shape(examples_per_session))

            examples_participant_training_sessions.append(examples_per_session)
            labels_participant_training_sessions.append(labels_per_session)
            print("@ traning sessions = ", np.shape(examples_participant_training_sessions))

        print('traning examples ', np.shape(examples_participant_training_sessions))
        print('traning labels ', np.shape(labels_participant_training_sessions))  
                
        examples_training_sessions_datasets = examples_participant_training_sessions
        labels_training_sessions_datasets = labels_participant_training_sessions
        
        # participants_num x days_per_session(10) x trail_per_day(4*num_participant) x #examples_window*#mov(26*22=572) x window_size x channel_num
        # participants_num x days_per_session(10) x trail_per_day(4*num_participant) x #examples_window*#mov(26*22=572)
        print('all traning examples ', np.shape(examples_training_sessions_datasets))
        print('all traning labels ', np.shape(labels_training_sessions_datasets))

    # store processed data to dictionary
    dataset_dictionnary = {"examples_training": np.array(examples_training_sessions_datasets, dtype=object),
                        "labels_training": np.array(labels_training_sessions_datasets, dtype=object)}
    return dataset_dictionnary

def read_data_training(path, store_path, number_of_gestures=22, number_of_cycles=4, window_size=50, 
                        size_non_overlap=10, num_participant=5, sessions_to_include=[0,1,2], 
                        switch=0,start_at_participant=1, include_gestures = None, spectrogram=False):
    """
    Wrapper for reading and processing raw data. A npy file containing a dirctory for processed data is stored.
    This directory has key `examples_training` that stores windows of processed emg signal and key `labels_training` 
    that stores corresponding labels

    Args:
        path: path to load training data
        sotre: path to store processed data npy file
        number_of_gestures: numer of gestures recorded 
        number_of_cycles: number of trials recorded for each motion
        window_size: analysis window size
        size_non_overlap: length of non-overlap portion between each analysis window
        num_participant: numer of participants to include; should be integer between 1~5
        sessions_to_include: array of integers defining which session to include in across day training
        swtich: determine which of the following case to run
                case 0 = across wearing locations; processed dataset will be in form participants_num x sessions_num(3) 
                case 1 = across subjects; processed dataset will be in form sessions_num(3) x participants_num;
                        when choosing case 1, remember to sepcify which subject will be used for base model 
                        in start_at_participant
                case 2 = across days (among the same subject and wearing location);
                    when choosing case 2, remember to sepcify which session to include in sessions_to_include;
                    one assumption for this case is that only one session (wearing location) is included for training
        start_at_participant: indicates which subject is the used as base model for DANN and SCADANN training 
                                in across subject training 
        include_gestures: list of gestures (in integer) to include in processed signals; include all gestures if None
        spectrogram: whether to process sEMG into spectrograms 

    """
    print("Loading and preparing Training datasets...")
    dataset_dictionnary = get_data_and_process_it_from_file(path=path, number_of_gestures=number_of_gestures,
                                                            number_of_cycles=number_of_cycles, window_size=window_size,
                                                            size_non_overlap=size_non_overlap, 
                                                            num_participant=num_participant,
                                                            sessions_to_include = sessions_to_include, 
                                                            switch=switch,
                                                            start_at_participant=start_at_participant,
                                                            include_gestures=include_gestures,
                                                            spectrogram=spectrogram)

    # store dictionary to pickle
    training_session_dataset_dictionnary = {}
    training_session_dataset_dictionnary["examples_training"] = dataset_dictionnary["examples_training"]
    training_session_dataset_dictionnary["labels_training"] = dataset_dictionnary["labels_training"]

    with open(store_path + "/training_session.pickle", 'wb') as f:
        pickle.dump(training_session_dataset_dictionnary, f, pickle.HIGHEST_PROTOCOL)


