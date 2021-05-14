'''
	Original by UlysseCoteAllard
		https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset

    Edited by Sonia Yuxiao Lai
'''

import os, sys
import pickle
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0,'../..')

from PrepareAndLoadData import feature_extraction

list_participant_training_1_to_skip = ["Participant0/Training1", "Participant0/Evaluation2", "Participant0/Evaluation3",
                                       "Participant2/Training1", "Participant2/Evaluation2", "Participant2/Evaluation3"]

def get_highest_average_emg_window(emg_signal, window_for_moving_average):
    """ For 3DC dataset only """
    max_average = 0.
    example = []
    for emg_vector in emg_signal:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= window_for_moving_average:
            example = example.transpose()
            example_filtered = []
            for channel in example:
                channel_filtered = feature_extraction.butter_bandpass_filter(channel, lowcut=20, highcut=495, fs=1000, order=4)
                # show_filtered_signal(channel, channel_filtered)
                example_filtered.append(channel_filtered)
            average = np.mean(np.abs(example_filtered))
            if average > max_average:
                max_average = average
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[1:]
    return max_average


def format_examples(emg_examples, feature_set_function, window_size=150, size_non_overlap=50):
    ''' 
    emg_examples: list of emg signals, each member represent one recording of a 10 channel emg
    
    feature_set_function:
        shape(formated_example) = (97, 10, 150)
        shape(featured_example) = (97, 385)
        might not need this for myo
    
    window_size, size_non_verlap: 150ms frame with 100ms overlap    time = freq * #example
        ignoring leftover examples at the end
        
    '''
    examples_to_calculate_features_set_from = []
    example = []
    for emg_vector in emg_examples:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))

        if len(example) >= window_size:
            # The example is of the shape TIME x CHANNEL. Make it of the shape CHANNEL x TIME
            example = example.transpose()
            # Go over each channel and bandpass filter it between 20 and 495 Hz.
            example_filtered = []
            for channel in example:
                channel_filtered = feature_extraction.butter_bandpass_filter(channel, lowcut=20, highcut=495, fs=1000, order=4)
                example_filtered.append(channel_filtered)
            # Add the filtered example to the list of examples to return and transpose the example array again to go
            # back to TIME x CHANNEL
            examples_to_calculate_features_set_from.append(example_filtered)
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]
    examples_features_set_calculated = feature_extraction.get_dataset_with_features_set(
        dataset=examples_to_calculate_features_set_from, features_set_function=feature_set_function)
    return examples_features_set_calculated

def read_files_to_format_training_session(path_folder_examples, feature_set_function,
                                          number_of_cycles, number_of_gestures, window_size,
                                          size_non_overlap):
    '''
    when #cycle = 1, examples are recorded at highest activation
    '''
    examples_training, labels_training = [], []
    # Check if the folder is empty, if so, skip it
    # path_folder_examples = path + "/" + folder_participant + "/" + training_directory + "/EMG"
    if len(os.listdir(path_folder_examples)) == 0:
        return [], [], []

    #--------  filename = `3dc_EMG_gesture_<#cycle>_<#gesture>.txt`-------------
    highest_activation_per_gesture = []
    for cycle in range(number_of_cycles):
        # path_folder_examples = path + folder_participant + "/" + training_directory + "/EMG"
        # This one instance, the participant only recorded one cycle of training. Skip it
        for participant_session_to_skip in list_participant_training_1_to_skip:
            if participant_session_to_skip in path_folder_examples:
                return [], [], []
        path_emg = path_folder_examples + "/3dc_EMG_gesture_%d_" % cycle
        examples, labels = [], []
        for gesture_index in range(number_of_gestures):
            examples_to_format = []

            for line in open(path_emg + '%d.txt' % gesture_index):
                #  strip() remove the "\n" character, split separate the data in a list. np.float
                #  transform each element of the list from a str to a float
                emg_signal = np.float32(line.strip().split(","))
                examples_to_format.append(emg_signal)
            
            if cycle == 1:  # This cycle is the second cycle and correspond to the highest effort baseline. Record it.
                if gesture_index == 0:
                    highest_activation_per_gesture.append(0)
                else:
                    highest_activation_per_gesture.append(get_highest_average_emg_window(
                        examples_to_format, window_for_moving_average=window_size))

            examples_formatted = format_examples(examples_to_format,
                                                 feature_set_function=feature_set_function, window_size=window_size,
                                                 size_non_overlap=size_non_overlap)
            examples.extend(examples_formatted)
            labels.extend(np.ones(len(examples_formatted)) * gesture_index)
        
        print("   SHAPE SESSION ", cycle, " EXAMPLES: ", np.shape(examples))
        examples_training.append(examples)
        labels_training.append(labels)
        print("   SHAPE ALL SESSION EXAMPLES: ", np.shape(examples_training))

    return examples_training, labels_training, highest_activation_per_gesture


def get_data_and_process_it_from_file(path, feature_set_function, number_of_gestures=11, number_of_cycles=4,
                                      window_size=150, size_non_overlap=50):
    examples_training_sessions_datasets, labels_training_sessions_datasets = [], []
    highest_activation_participants = []
    examples_evaluation_sessions_datasets, labels_evaluation_sessions_datasets, timestamps_emg_evaluation = [], [], []
    angles_with_timestamps_emg_evaluation = []

    training_datetimes, evaluation_datetimes = [], []
    # for index_participant in range(22):
    for index_participant in range(1):
        # Those two participant did not complete the experiment
        if index_participant != 10 and index_participant != 11:
            folder_participant = "/Participant" + str(index_participant)
            sessions_directories = os.listdir(path + folder_participant)

            examples_participant_training_sessions, labels_participant_training_sessions = [], []
            highest_activation_per_session = []
            examples_participant_evaluation_sessions, labels_participant_evaluation_sessions = [], []
            timestamps_evaluation_participant_sessions, angles_with_timestamps_participant_sessions = [], []
            
            """
            Don't need Evaluation part for now, only process Training 
            """
            for session_directory in sessions_directories:
                print("current dir = ", session_directory)

                if "Training" in session_directory:
                    path_folder_examples = path + "/" + folder_participant + "/" + session_directory + "/EMG"
                    examples_training, labels_training, highest_activation_per_gesture = \
                        read_files_to_format_training_session(path_folder_examples=path_folder_examples,
                                                              number_of_cycles=number_of_cycles,
                                                              number_of_gestures=number_of_gestures,
                                                              window_size=window_size,
                                                              size_non_overlap=size_non_overlap,
                                                              feature_set_function=feature_set_function)
                    if len(examples_training) > 0:
                        # These instances, the participant only recorded one cycle of training. Skip it
                        skip_it = False
                        for participant_session_to_skip in list_participant_training_1_to_skip:
                            if participant_session_to_skip in path_folder_examples:
                                skip_it = True
                        if skip_it is False:
                            examples_participant_training_sessions.append(examples_training)
                            labels_participant_training_sessions.append(labels_training)
                            highest_activation_per_session.append(highest_activation_per_gesture)
                            print("@ traning sessions = ", np.shape(examples_participant_training_sessions))
            
            # store processed data to dictionary
            print('traning examples ', np.shape(examples_participant_training_sessions))
            examples_training_sessions_datasets.append(examples_participant_training_sessions)
            print('all traning examples ', np.shape(examples_training_sessions_datasets))
            
            print('traning labels ', np.shape(labels_participant_training_sessions))
            labels_training_sessions_datasets.append(labels_participant_training_sessions)
            print('all traning labels ', np.shape(labels_training_sessions_datasets))

    dataset_dictionnary = {"examples_training": np.array(examples_training_sessions_datasets, dtype=object),
                           "labels_training": np.array(labels_training_sessions_datasets, dtype=object)}

    return dataset_dictionnary


def read_data_training(path, store_path, features_set_name, feature_set_function, number_of_gestures=11, number_of_cycles=4,
                       window_size=150, size_non_overlap=50):
    print("Loading and preparing Training datasets...")
    dataset_dictionnary = get_data_and_process_it_from_file(path=path, number_of_gestures=number_of_gestures,
                                                            number_of_cycles=number_of_cycles, window_size=window_size,
                                                            size_non_overlap=size_non_overlap,
                                                            feature_set_function=feature_set_function)

    training_session_dataset_dictionnary = {}
    training_session_dataset_dictionnary["examples_training"] = dataset_dictionnary["examples_training"]
    training_session_dataset_dictionnary["labels_training"] = dataset_dictionnary["labels_training"]

    with open(f"{store_path}/{features_set_name}_training_session.pickle", 'wb') as f:
        pickle.dump(training_session_dataset_dictionnary, f, pickle.HIGHEST_PROTOCOL)

