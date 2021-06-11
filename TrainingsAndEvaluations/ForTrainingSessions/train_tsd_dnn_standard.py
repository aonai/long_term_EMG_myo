import os, sys
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from Models.TSD_neural_network import TSD_Network
from PrepareAndLoadData.load_dataset_in_dataloader import load_dataloaders_training_sessions

def train_model_standard(model, criterion, optimizer, scheduler, dataloaders, num_epochs=500, precision=1e-8,
                         patience=10, patience_increase=10):
    """
    Return best_state, or weights of trained model
    """
    since = time.time()

    best_loss = float('inf')

    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data

                inputs, labels = inputs, labels
                # zero the parameter gradients
                optimizer.zero_grad()
                if phase == 'train':
                    model.train()
                    # forward
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()

                else:
                    model.eval()
                    with torch.no_grad():

                        # forward
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs.data, 1)

                        loss = criterion(outputs, labels)
                        loss = loss.item()

                # statistics
                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            print('{} Loss: {:.8f} Acc: {:.8}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience:
            break
    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state

def load_checkpoint(model, filename, optimizer=None, scheduler=None, strict=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if optimizer is not None:
            print("Loading Optimizer")
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch

def train_fine_tuning(examples_datasets_train, labels_datasets_train, num_kernels,
                    number_of_cycles_total=40, 
                    path_weight_to_save_to="Weights/unknown", number_of_classes=22, batch_size=128,
                    feature_vector_input_length=252, learning_rate=0.002515):
    """
    Train model using the first session, then fine tune using the others. 

    Args:
        examples_datasets: ndarray of input examples
        labels_datasets: ndarray of labels for each example
        num_kernels (list of integer): each integer is width of TSD linear block of corresponding layer 
        number_of_cycle_for_first_training:  number of total trails for the first training session
        number_of_cycles_rest_of_training:  number of total trails for the rest
        path_weight_to_save_to: where to save trained model
        number_of_classes: number of classes to train
        batch_size:  size of one batch in dataloader
        feature_vector_input_length: length of one example data (252)
        learning_rate
    """ 
    participants_train, participants_validation, _ = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycles_total=number_of_cycles_total)

    print("START TRAINING")
    for participant_i in range(len(participants_train)):
        print("Participant: ", participant_i)
        for session_j in range(0, len(participants_train[participant_i])):
            print("Session: ", session_j)
            # Define Model
            model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                feature_vector_input_length=feature_vector_input_length)

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean')

            # Define Optimizer
            print(model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-4
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                             verbose=True, eps=precision)

            if session_j > 0:
                # Fine-tune from the previous training
                model, _, _, start_epoch = load_checkpoint(
                    model=model, optimizer=None, scheduler=None,
                    filename=path_weight_to_save_to +
                             "/participant_%d/best_state_%d.pt" %
                             (participant_i, session_j - 1))

            best_state = train_model_standard(model=model, criterion=cross_entropy_loss_classes, optimizer=optimizer,
                                              scheduler=scheduler,
                                              dataloaders={"train": participants_train[participant_i][session_j],
                                                           "val": participants_validation[participant_i][session_j]},
                                              precision=precision, patience=10, patience_increase=10)

            if not os.path.exists(path_weight_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weight_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_state, f=path_weight_to_save_to +
                                     "/participant_%d/best_state_%d.pt"
                                     % (participant_i, session_j))


def test_TSD_DNN_on_training_sessions(examples_datasets_train, labels_datasets_train, num_neurons,
                                      feature_vector_input_length=252,
                                      path_weights='/Weights', save_path='results', algo_name="Normal_Training",
                                      use_only_first_training=False, cycle_for_test=None, number_of_cycles_total=40,
                                      number_of_classes=22, across_sub=False):
    """
    Test trained model. Stores a txt and npy files that include predictions, ground truths, accuracy table, and 
    overall accuracies. 

    Args: 
        examples_datasets: ndarray of input examples
        labels_datasets: ndarray of labels for each example
        num_neurons (list of integer): each integer is width of TSD linear block of corresponding layer 
        path_weights: where to load trained model
        save_path: where to save test results
        algo_name: nickname of model (this will be included in file name of test results)
        use_only_first_training: 
            load weights from first trainning session only if True; 
            otherwise use fine tunning weights (trained from all four sessions)
        cycle_for_test: which session to use for testing
        number_of_cycles_total: total number of trails in one session
        number_of_classes: number of classes to train

    """
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                                                number_of_cycles_total=number_of_cycles_total,
                                                                 batch_size=128*3, cycle_for_test=cycle_for_test)
    model_outputs = []
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        model_outputs_participant = []
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        
        # get model
        model = TSD_Network(number_of_class=number_of_classes, feature_vector_input_length=feature_vector_input_length,
                            num_neurons=num_neurons)
        
        # start test
        for session_index, training_session_test_data in enumerate(dataset_test):
            print(session_index, " SESSION   data = ", len(training_session_test_data.dataset))
            if across_sub:
                best_state = torch.load(
                    path_weights + "/participant_%d/best_state_%d.pt" %
                    (0, 0))
            else:
                if use_only_first_training:
                    best_state = torch.load(
                        path_weights + "/participant_%d/best_state_%d.pt" %
                        (participant_index, 0))
                else:
                    best_state = torch.load(
                        path_weights + "/participant_%d/best_state_%d.pt" %
                        (participant_index, session_index))
            best_weights = best_state['state_dict']
            
            # load trained model
            model.load_state_dict(best_weights)
            
            # compare prediction and ground truth
            predictions_training_session = []
            ground_truth_training_sesssion = []
            model_outputs_session = []
            with torch.no_grad():
                # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
                    predictions_training_session.extend(predicted.cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
                    # expected batch_size = 3*128 = 384
                    # print("   one batch ", output.shape)
            print("Participant: ", participant_index, " Accuracy: ",
                np.mean(np.array(predictions_training_session) == np.array(ground_truth_training_sesssion)))
            predictions_participant.append(predictions_training_session)
            model_outputs_participant.append(model_outputs_session)
            ground_truth_participant.append(ground_truth_training_sesssion)
            accuracies_participant.append(np.mean(np.array(predictions_training_session) ==
                                                np.array(ground_truth_training_sesssion)))
            # print("ground truth ", np.shape(ground_truth_training_sesssion))
            # print("ground_truth_participant ", np.shape(ground_truth_participant))

        accuracies.append(np.array(accuracies_participant))
        predictions.append(predictions_participant)
        model_outputs.append(model_outputs_participant)
        ground_truths.append(ground_truth_participant)
        print("ACCURACY PARTICIPANT ", participant_index, ": ", accuracies_participant)

    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies):
        accuracies_to_display.append(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))
        
    filter_size = num_neurons
    if use_only_first_training:
        file_to_open = save_path + '/' + algo_name + "_no_retraining.txt"
        np.save(save_path + "/predictions_" + algo_name + "_no_retraining", np.array((accuracies_to_display, 
                                                                            ground_truths, 
                                                                            predictions, 
                                                                            model_outputs), dtype=object))
    else:
        file_to_open = save_path + '/' + algo_name + "_WITH_RETRAINING_" + str(filter_size[1]) + ".txt"
        np.save(save_path + "/predictions_" + algo_name + "_no_retraining", np.array((accuracies_to_display, 
                                                                            ground_truths, 
                                                                            predictions, 
                                                                            model_outputs), dtype=object))
    with open(file_to_open, "a") as \
            myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truths) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))


