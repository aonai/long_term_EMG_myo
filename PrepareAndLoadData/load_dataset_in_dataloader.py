import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset


def get_dataloader(examples_datasets, labels_datasets, number_of_cycle_for_first_training=40,
                   number_of_cycles_rest_of_training=40, batch_size=128,
                   drop_last=True, shuffle=True,
                   number_of_cycles_total=40, validation_set_ratio=0.1, get_validation_set=True, cycle_for_test=None):
    """
    Put examples and labels into an array of datalaoders (train, valid, and test)
    Test dataloader only include data whose trail numbers are 4n

    Args:
        examples_datasets: ndarray of input examples
        labels_datasets: ndarray of labels for each example
        number_of_cycle_for_first_training:  number of total trails for the first training session
        number_of_cycles_rest_of_training:  number of total trails for the rest
        batch_size: size of one batch in dataloader
        drop_last: whether to drop data when last set of data < batch_size (only used for train dataloader)
        shuffle: whether to shuffle
        number_of_cycles_total: number of trails performed for each session (assuming that all session have the same trail size)
        validation_set_ratio: ratio of validation set out of entire dataset, the rest are training set 
        get_validation_set: whether to return validation dataloader
        cycle_for_test: specify which cycle will be used for testing; no test dataloader returned if None
    
    Returns: 
        training, validation, and test dataloaders in shape (num_participants x num_sessions)
            expected total shape of each dataloader =  
                    (num_trails*num_examples_window*num_mov(40*26*22=22880 total) x features_windwo(252))
                        num_trails = 0.9*total for training 
                                   = 0.1*total for validation 
                                   = 0.25*total for testing 
    """
    participants_dataloaders, participants_dataloaders_validation, participants_dataloaders_test = [], [], []

    for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
        print("GET one participant_examples ", np.shape(participant_examples))
        dataloaders_trainings = []
        dataloaders_validations = []
        dataloaders_testing = []
        
        k = 0
        # X = signals, Y = labels
        for training_index_examples, training_index_labels in zip(participant_examples, participant_labels):
            print("   GET one training_index_examples ", np.shape(training_index_examples), " at ", k)
            cycles_to_add_to_train = number_of_cycle_for_first_training
            if k > 0:
                cycles_to_add_to_train = number_of_cycles_rest_of_training
                
            X_associated_with_training_i, Y_associated_with_training_i = [], []
            X_test_associated_with_training_i, Y_test_associated_with_training_i = [], []
            for cycle in range(cycles_to_add_to_train):
                # print("cycle = ", cycle)
                examples_cycles = training_index_examples[cycle]
                labels_cycles = training_index_labels[cycle]

                # print("      GET one examples_cycles ", np.shape(examples_cycles), " at ", cycle)
                # print("      GET one labels_cycles ", np.shape(labels_cycles), " at ", cycle)
                if cycle < cycles_to_add_to_train:
                    X_associated_with_training_i.extend(examples_cycles)
                    Y_associated_with_training_i.extend(labels_cycles)
                if cycle_for_test is not None and cycle_for_test == (cycle%4):
                    # print("      save test cycles ", cycle, "---", cycle_for_test)
                    X_test_associated_with_training_i.extend(examples_cycles)
                    Y_test_associated_with_training_i.extend(labels_cycles)

            print("   GOT one group XY ", np.shape(X_associated_with_training_i), "  ",  np.shape(Y_associated_with_training_i))
            print("       one group XY test ", np.shape(X_test_associated_with_training_i), "  ",  np.shape(X_test_associated_with_training_i))
            k += 1

            if get_validation_set:
                # Shuffle X and Y and separate them in a train and validation set.
                X, X_valid, Y, Y_valid = train_test_split(X_associated_with_training_i, Y_associated_with_training_i,
                                                          test_size=validation_set_ratio, shuffle=True)
                print("       one group XY train", np.shape(X), "  ",  np.shape(Y))
                print("       one group XY valid", np.shape(X_valid), "  ",  np.shape(X_valid))
            else:
                X, Y = X_associated_with_training_i, Y_associated_with_training_i
                print("       one group XY train", np.shape(X), "  ",  np.shape(Y))
            
            # trainning dataloader
            train = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
                                  torch.from_numpy(np.array(Y, dtype=np.int64)))
            trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle,
                                                      drop_last=drop_last)
            dataloaders_trainings.append(trainloader)
            
            # validation dataloader
            if get_validation_set:
                validation = TensorDataset(torch.from_numpy(np.array(X_valid, dtype=np.float32)),
                                           torch.from_numpy(np.array(Y_valid, dtype=np.int64)))
                validationloader = torch.utils.data.DataLoader(validation, batch_size=len(X_valid), shuffle=shuffle,
                                                               drop_last=False)
                dataloaders_validations.append(validationloader)

            # testing dataloader 
            if len(X_test_associated_with_training_i) > 0:
                test = TensorDataset(torch.from_numpy(np.array(X_test_associated_with_training_i, dtype=np.float32)),
                                     torch.from_numpy(np.array(Y_test_associated_with_training_i, dtype=np.int64)))
                testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                                         drop_last=False)
                dataloaders_testing.append(testLoader)

        participants_dataloaders.append(dataloaders_trainings)
        if get_validation_set:
            participants_dataloaders_validation.append(dataloaders_validations)
        participants_dataloaders_test.append(dataloaders_testing)
        print("dataloaders: ")
        print("   train ", np.shape(participants_dataloaders))
        print("   valid ", np.shape(participants_dataloaders_validation))
        print("   test ", np.shape(participants_dataloaders_test))
    return participants_dataloaders, participants_dataloaders_validation, participants_dataloaders_test

def load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                       number_of_cycle_for_first_training=40, number_of_cycles_rest_of_training=40,
                                       number_of_cycles_total=40, 
                                       batch_size=128, drop_last=True, shuffle=True, get_validation_set=True,
                                       cycle_for_test=None):
    """
    Wrapper for building dataloaders. 
    
    Args:
        examples_datasets_train: ndarray of input examples
        labels_datasets_train: ndarray of labels for each example
        number_of_cycle_for_first_training:  number of total trails for the first training session
        number_of_cycles_rest_of_training:  number of total trails for the rest
        number_of_cycles_total: number of trails performed for each session (assuming that all session have the same trail size)
        batch_size: size of one batch in dataloader
        drop_last: whether to drop data when last set of data < batch_size (only used for train dataloader)
        shuffle: whether to shuffle
        get_validation_set: whether to return validation dataloader
        cycle_for_test: specify which cycle will be used for testing; no test dataloader returned if None
    
    Returns: 
        training, validatio, and test dataloaders in shape (num_participants x num_sessions)
    """
    train, validation, test = get_dataloader(examples_datasets_train, labels_datasets_train,
                                             number_of_cycle_for_first_training=number_of_cycle_for_first_training,
                                             number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, 
                                             number_of_cycles_total = number_of_cycles_total, 
                                             batch_size=batch_size,
                                             drop_last=drop_last, shuffle=shuffle,
                                             get_validation_set=get_validation_set,
                                             cycle_for_test=cycle_for_test)

    return train, validation, test
