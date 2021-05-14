'''
	Original by UlysseCoteAllard
		https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset

    Edited by Sonia Yuxiao Lai
'''

import os, sys
import pickle
import numpy as np
import pandas as pd

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


def format_examples(emg_examples, window_size=50, size_non_overlap=10):
    """ 
    emg_examples: list of emg signals, each row represent one recording of a 8 channel emg
    feature_set_function
    window_size: analysis window size
    size_non_overlap: length of non-overlap portion between each analysis window
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
            formated_examples.append(example.copy())
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
    for index_participant in range(1,2):
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


