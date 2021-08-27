import os
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from Models.spectrogram_ConvNet import SpectrogramConvNet
from Models.TSD_neural_network import TSD_Network
from PrepareAndLoadData.load_dataset_in_dataloader import load_dataloaders_training_sessions
from TrainingsAndEvaluations.ForTrainingSessions.train_tsd_dnn_standard import load_checkpoint

def DANN_Training(gesture_classifier, crossEntropyLoss, optimizer_classifier, train_dataset_source, scheduler,
                     train_dataset_target, validation_dataset_source, patience_increment=10, max_epochs=500,
                     domain_loss_weight=1e-1):
    """
    Args:
        gesture_classification: pre-trained TSD model
        train_dataset_source: the first session of a participant's training set
        train_dataset_target: one seesion (except for the first) of a participant's traning set
        validation_dataset_source:  the first session of a participant's validation set
        patience_increment: number of epchos to wait after no best loss is found and before existing training
        target: unlabeled; source: labeled
        domain_loss_weight: coefficient of doman loss percantage to account in calculating loss 
                        (loss_main_source and loss_doman_target)
    Returns:
        best_state: trained weights 
    """
    since = time.time()
    patience = 0 + patience_increment

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(2):
        state_dict = gesture_classifier.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batchNorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    best_loss = float("inf")
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}

    print("STARTING TRAINING")
    for epoch in range(1, max_epochs):
        epoch_start = time.time()

        loss_main_sum, n_total = 0, 0
        loss_domain_sum, loss_src_class_sum, loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
        running_corrects, running_correct_domain, total_for_accuracy, total_for_domain_accuracy = 0, 0, 0, 0

        'TRAINING'
        gesture_classifier.train()
        for source_batch, target_batch in zip(train_dataset_source, train_dataset_target):

            input_source, labels_source = source_batch
            input_source, labels_source = input_source, labels_source
            input_target, _ = target_batch
            input_target = input_target

            # Feed the inputs to the classifier network
            # Retrieves the BN weights calculated so far for the source dataset
            BN_weights = list_dictionaries_BN_weights[0]
            gesture_classifier.load_state_dict(BN_weights, strict=False)
            pred_gesture_source, pred_domain_source = gesture_classifier(input_source, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised/self-supervised gesture classification
            loss_source_class = crossEntropyLoss(pred_gesture_source, labels_source)

            # Try to be bad at the domain discrimination for the full network

            label_source_domain = torch.zeros(len(pred_domain_source), device='cpu', dtype=torch.long)
            loss_domain_source = crossEntropyLoss(pred_domain_source, label_source_domain)
            # Combine all the loss of the classifier
            loss_main_source = (0.5 * loss_source_class + domain_loss_weight * loss_domain_source)

            ' Update networks '
            # Update classifiers.
            # Zero the gradients
            optimizer_classifier.zero_grad()
            # loss_main_source.backward(retain_graph=True)
            loss_main_source.backward()
            optimizer_classifier.step()
            # Save the BN stats for the source
            state_dict = gesture_classifier.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)

            _, pred_domain_target = gesture_classifier(input_target, get_all_tasks_output=True)
            label_target_domain = torch.ones(len(pred_domain_target), device='cpu', dtype=torch.long)
            loss_domain_target = 0.5 * (crossEntropyLoss(pred_domain_target, label_target_domain))
            # Combine all the loss of the classifier
            loss_domain_target = 0.5 * domain_loss_weight * loss_domain_target
            # Update classifiers.
            # Zero the gradients
            loss_domain_target.backward()
            optimizer_classifier.step()

            # Save the BN stats for the target
            state_dict = gesture_classifier.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[1] = copy.deepcopy(batch_norm_dict)

            loss_main = loss_main_source + loss_domain_target
            loss_domain = loss_domain_source + loss_domain_target

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(pred_gesture_source.data, 1)
            running_corrects += torch.sum(gestures_predictions_source == labels_source.data)
            total_for_accuracy += labels_source.size(0)

            _, gestures_predictions_domain_source = torch.max(pred_domain_source.data, 1)
            _, gestures_predictions_domain_target = torch.max(pred_domain_target.data, 1)
            running_correct_domain += torch.sum(gestures_predictions_domain_source == label_source_domain.data)
            running_correct_domain += torch.sum(gestures_predictions_domain_target == label_target_domain.data)
            total_for_domain_accuracy += label_source_domain.size(0)
            total_for_domain_accuracy += label_target_domain.size(0)

        print('Accuracy source %4f,'
              ' main loss classifier %4f,'
              ' source classification loss %4f,'
              ' loss domain distinction %4f,'
              ' accuracy domain distinction %4f'
              %
              (running_corrects.item() / total_for_accuracy,
               loss_main_sum / n_total,
               loss_src_class_sum / n_total,
               loss_domain_sum / n_total,
               running_correct_domain.item() / total_for_domain_accuracy
               ))

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_val = 0

        # BN_weights = copy.deepcopy(list_dictionaries_BN_weights[0])
        # gesture_classifier.load_state_dict(BN_weights, strict=False)
        gesture_classifier.eval()
        for validation_batch in validation_dataset_source:
            # get the inputs
            inputs, labels = validation_batch

            inputs, labels = inputs, labels
            # zero the parameter gradients
            optimizer_classifier.zero_grad()

            with torch.no_grad():
                # forward
                outputs = gesture_classifier(inputs)
                _, predictions = torch.max(outputs.data, 1)

                loss = crossEntropyLoss(outputs, labels)
                loss = loss.item()

                # statistics
                running_loss_validation += loss
                running_corrects_validation += torch.sum(predictions == labels.data)
                total_validation += labels.size(0)
                n_total_val += 1

        epoch_loss = running_loss_validation / n_total_val
        epoch_acc = running_corrects_validation.item() / total_validation
        print('{} Loss: {:.8f} Acc: {:.8}'.format("VALIDATION", epoch_loss, epoch_acc))

        scheduler.step(running_loss_validation / n_total_val)
        if running_loss_validation / n_total_val < best_loss:
            print("New best validation loss: ", running_loss_validation / n_total_val)
            best_loss = running_loss_validation / n_total_val
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            gesture_classifier.load_state_dict(BN_weights, strict=False)
            best_state = {'epoch': epoch, 'state_dict': copy.deepcopy(gesture_classifier.state_dict()),
                          'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
            patience = epoch + patience_increment

        if patience < epoch:
            break

        print("Epoch {} of {} took {:.3f}s".format(
            epoch, max_epochs, time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_state


def train_DANN(examples_datasets_train, labels_datasets_train, num_kernels, 
                          path_weights_to_save_to="Weights_TSD/DANN", batch_size=512, patience_increment=10,
                          path_weights_fine_tuning="Weights_TSD/TSD",
                          number_of_cycles_total=40, number_of_cycle_for_first_training=40, number_of_classes=22, 
                          feature_vector_input_length=252, learning_rate=0.002515,neural_net="TSD", filter_size=(4, 10)):
    """
    Wrapper for trainning and saving a DANN model. 

    Args: 
        examples_datasets_train: ndarray of input examples
        labels_datasets_train: ndarray of labels for each example
        num_neurons (list of integer): each integer is width of TSD linear block of corresponding layer or output channels of ConvNet model
        path_weights_to_save_to: where to save trained model
        path_weights_fine_tuning: path to load normal TSD_DNN or ConvNet weights 
        batch_size:  size of one batch in dataloader
        number_of_cycles_total: total number of trails in one session
        number_of_cycle_for_first_training: total number of trials in the first session
        number_of_classes: number of classes to train
        feature_vector_input_length: length of one example data (252) for TSD model
        learning_rate
        neural_net: specify which training model to use; options are 'TSD' and 'ConvNet'
        filter_size: kernel size of ConvNet model; should be a 2d list of shape (m, 2), where m (4) is number of levels
    """
    participants_train, participants_validation, participants_test = load_dataloaders_training_sessions(
        examples_datasets_train, labels_datasets_train, batch_size=batch_size,
        number_of_cycles_total=number_of_cycles_total, 
        number_of_cycles_rest_of_training=number_of_cycles_total,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        get_validation_set=True)

    for participant_i in range(len(participants_train)):
        print("SHAPE SESSIONS: ", np.shape(participants_train[participant_i]))

        # Skip the first session as it will be identical to normal training
        for session_j in range(1, len(participants_train[participant_i])):
            print(np.shape(participants_train[participant_i][session_j]))

            if neural_net == 'TSD':
                gesture_classification = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                feature_vector_input_length=feature_vector_input_length)
            elif neural_net == 'Spectrogram':
                gesture_classification = SpectrogramConvNet(number_of_class=number_of_classes, num_kernels=num_kernels,
                                kernel_size=filter_size)

            # loss functions
            crossEntropyLoss = nn.CrossEntropyLoss()
            # optimizer
            precision = 1e-8
            optimizer_classifier = optim.Adam(gesture_classification.parameters(), lr=learning_rate, betas=(0.5, 0.999))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_classifier, mode='min', factor=.2,
                                                             patience=5, verbose=True, eps=precision)
            # Fine-tune from the previous training
            gesture_classification, optimizer_classifier, scheduler, start_epoch = load_checkpoint( 
                model=gesture_classification, optimizer=optimizer_classifier, scheduler=scheduler,  
                filename=path_weights_fine_tuning + "/participant_%d/best_state_%d.pt" % (participant_i, 0))  

            best_weights = DANN_Training(gesture_classifier=gesture_classification, scheduler=scheduler, 
                                            optimizer_classifier=optimizer_classifier,  
                                            train_dataset_source=participants_train[participant_i][0],  
                                            train_dataset_target=participants_train[participant_i][session_j],  
                                            validation_dataset_source=participants_validation[participant_i][0],  
                                            crossEntropyLoss=crossEntropyLoss,  
                                            patience_increment=patience_increment,  
                                            domain_loss_weight=1e-1)

            if not os.path.exists(path_weights_to_save_to  + "/participant_%d" % participant_i):
                os.makedirs(path_weights_to_save_to + "/participant_%d" % participant_i)
            torch.save(best_weights, f=path_weights_to_save_to + "/participant_%d/best_state_%d.pt" % (participant_i, session_j))

def test_DANN_on_training_sessions(examples_datasets_train, labels_datasets_train, num_neurons, feature_vector_input_length=252,
                              path_weights_normal='/Weights/TSD', path_weights_DA='/Weights/DANN', algo_name="DANN",
                              save_path='results', number_of_cycles_total=40, number_of_cycle_for_first_training=40,
                              cycle_for_test=None, number_of_classes=22,
                              across_sub=False, neural_net="TSD", filter_size=(4, 10)):
    """
    Test trained model. Stores a txt and npy files that include accuracies, predictions, ground truths, and model outputs.
    overall accuracies.

    Args: 
        examples_datasets_train: ndarray of input examples
        labels_datasets_train: ndarray of labels for each example
        num_neurons (list of integer): each integer is width of TSD linear block of corresponding layer or output channels of ConvNet model
        feature_vector_input_length: size of one example (=252)
        path_weights_normal: where to load trained TSD or ConvNet model
        path_weights_DA: where to load trained DANN model
        algo_name: nickname of model (this will be included in file name of test results)
        save_path: where to save test results
        number_of_cycles_total: total number of trails in one session
        number_of_cycle_for_first_training: total number of trials in the first session
        cycle_for_test: which session to use for testing
        number_of_classes: number of classes to train
        across_sub: whether model is trained across subject; base model weights is always 
                    stored at /participant_0/best_state_0
        neural_net: specify which training model to use; options are 'TSD' and 'ConvNet'
        filter_size: kernel size of ConvNet model; should be a 2d list of shape (m, 2), where m (4) is number of levels

    """
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                                                number_of_cycles_total=number_of_cycles_total,
                                                                number_of_cycles_rest_of_training=number_of_cycles_total,
                                                                number_of_cycle_for_first_training=number_of_cycle_for_first_training,
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

        if neural_net == 'TSD':
            model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_neurons,
                            feature_vector_input_length=feature_vector_input_length)
        elif neural_net == 'Spectrogram':
            model = SpectrogramConvNet(number_of_class=number_of_classes, num_kernels=num_neurons,
                            kernel_size=filter_size)
        
        print(np.shape(dataset_test))
        for session_index, training_session_test_data in enumerate(dataset_test):
            if across_sub:
                best_state = torch.load(
                    path_weights_DA + "/participant_%d/best_state_%d.pt" %
                    (0, 1))
            else:
                if session_index == 0:
                    best_state = torch.load(
                        path_weights_normal + "/participant_%d/best_state_%d.pt" %
                        (participant_index, 0))
                else:
                    best_state = torch.load(
                        path_weights_DA + "/participant_%d/best_state_%d.pt" %
                        (participant_index, session_index))  # There is 2 evaluation sessions per training
            best_weights = best_state['state_dict']
            model.load_state_dict(best_weights)

            model_outputs_session = []
            predictions_training_session = []
            ground_truth_training_sesssion = []
            with torch.no_grad():
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
                    predictions_training_session.extend(predicted.cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
            print("Participant ID: ", participant_index, " Session ID: ", session_index, " Accuracy: ",
                  np.mean(np.array(predictions_training_session) == np.array(ground_truth_training_sesssion)))
            predictions_participant.append(predictions_training_session)
            model_outputs_participant.append(model_outputs_session)
            ground_truth_participant.append(ground_truth_training_sesssion)
            accuracies_participant.append(np.mean(np.array(predictions_training_session) ==
                                                  np.array(ground_truth_training_sesssion)))
        accuracies.append(np.array(accuracies_participant))
        predictions.append(predictions_participant)
        model_outputs.append(model_outputs_participant)
        ground_truths.append(ground_truth_participant)
        print("ACCURACY PARTICIPANT: ", accuracies_participant)
    print(np.array(accuracies))
    accuracies_to_display = []
    for accuracies_from_participant in np.array(accuracies):
        # accuracies_to_display.extend(accuracies_from_participant)
        accuracies_to_display.append(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))
    
    file_to_open = save_path + '/' + algo_name + ".txt"
    np.save(save_path + "/predictions_" + algo_name, np.array((accuracies_to_display, 
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

