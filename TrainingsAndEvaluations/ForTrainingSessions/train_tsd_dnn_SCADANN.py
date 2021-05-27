import os
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from Models.TSD_neural_network import TSD_Network
from PrepareAndLoadData.load_dataset_in_dataloader import load_dataloaders_training_sessions
from TrainingsAndEvaluations.ForTrainingSessions.train_tsd_dnn_standard import load_checkpoint
from TrainingsAndEvaluations.ForTrainingSessions.SCADANN_utils import generate_dataloaders_for_SCADANN


def SCADANN_BN_training(replay_dataset_train, target_validation_dataset, target_dataset, model, crossEntropyLoss,
                        optimizer_classifier, scheduler, patience_increment=10, max_epochs=500,
                        domain_loss_weight=2e-1):
    """
    replay_dataset_train: train dataset of previous 
    target_validation_dataset: validation dataset of current session
    target_dataset: train dataset of current session
    model: trained DANN model for current session
    crossEntropyLoss
    optimizer_classifier
    scheduler
    patience_increment: number of epchos to wait after no best loss is found and before existing training
    max_epochs
    domain_loss_weight: weight of doman loss in calculating source and target losses
        source = previous datasets, target = current dataset
    """
    since = time.time()

    # Create a list of dictionaries that will hold the weights of the batch normalisation layers for each dataset
    #  (i.e. each participants)
    list_dictionaries_BN_weights = []
    for index_BN_weights in range(2):
        state_dict = model.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "batchNorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

    patience = 0 + patience_increment

    best_loss = float("inf")
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(),
                  'scheduler': scheduler.state_dict()}

    print("STARTING TRAINING")
    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        n_total = 0
        loss_main_sum, loss_domain_sum, loss_src_class_sum, loss_target_class_sum = 0, 0, 0, 0
        running_corrects_source, running_corrects_target, running_correct_domain = 0, 0, 0
        total_for_accuracy_source, total_for_accuracy_target, total_for_domain_accuracy = 0, 0, 0

        'TRAINING'
        model.train()
        # alpha = 1e-1*(1/epoch)+1e-1
        alpha = 0.
        for source_batch, target_batch in zip(replay_dataset_train, target_dataset):
            input_source, labels_source = source_batch
            input_source, labels_source = input_source, labels_source
            input_target, labels_target = target_batch
            input_target, labels_target = input_target, labels_target

            # Feed the inputs to the classifier network
            # Retrieves the BN weights calculated so far for the source dataset
            BN_weights = list_dictionaries_BN_weights[0]
            model.load_state_dict(BN_weights, strict=False)
            pred_source, pred_domain_source = model(input_source, get_all_tasks_output=True)

            'Classifier losses setup.'
            # Supervised/self-supervised gesture classification
            loss_source_class = crossEntropyLoss(pred_source, labels_source)

            # Try to be bad at the domain discrimination for the full network

            label_source_domain = torch.zeros(len(pred_domain_source), device='cpu', dtype=torch.long)
            loss_domain_source = ((1 - alpha) * crossEntropyLoss(pred_domain_source, label_source_domain))
            # Combine all the loss of the classifier
            loss_main_source = (0.5 * loss_source_class + domain_loss_weight * loss_domain_source)

            ' Update networks '
            # Update classifiers.
            # Zero the gradients
            optimizer_classifier.zero_grad()
            #loss_main_source.backward(retain_graph=True)
            loss_main_source.backward()
            optimizer_classifier.step()
            # Save the BN stats for the source
            state_dict = model.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[0] = copy.deepcopy(batch_norm_dict)
            # Load the BN statistics for the target
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            model.load_state_dict(BN_weights, strict=False)
            model.train()

            pred_target, pred_domain_target = model(input_target, get_all_tasks_output=True)
            loss_target_class = crossEntropyLoss(pred_target, labels_target)
            label_target_domain = torch.ones(len(pred_domain_target), device='cpu', dtype=torch.long)
            loss_domain_target = 0.5 * (crossEntropyLoss(pred_domain_target, label_target_domain))
            # Combine all the loss of the classifier
            loss_main_target = (0.5 * loss_target_class + domain_loss_weight * loss_domain_target)
            # Update classifiers.
            # Zero the gradients
            loss_main_target.backward()
            optimizer_classifier.step()

            # Save the BN stats for the target
            state_dict = model.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights[1] = copy.deepcopy(batch_norm_dict)

            loss_main = loss_main_source + loss_main_target
            loss_domain = loss_domain_source + loss_domain_target

            loss_domain_sum += loss_domain.item()
            loss_src_class_sum += loss_source_class.item()
            loss_target_class_sum += loss_target_class.item()
            loss_target_class += loss_target_class.item()
            loss_main_sum += loss_main.item()
            n_total += 1

            _, gestures_predictions_source = torch.max(pred_source.data, 1)
            running_corrects_source += torch.sum(gestures_predictions_source == labels_source.data)
            total_for_accuracy_source += labels_source.size(0)

            _, gestures_predictions_target = torch.max(pred_target.data, 1)
            running_corrects_target += torch.sum(gestures_predictions_target == labels_target.data)
            total_for_accuracy_target += labels_target.size(0)

            _, gestures_predictions_domain_source = torch.max(pred_domain_source.data, 1)
            _, gestures_predictions_domain_target = torch.max(pred_domain_target.data, 1)
            running_correct_domain += torch.sum(gestures_predictions_domain_source == label_source_domain.data)
            running_correct_domain += torch.sum(gestures_predictions_domain_target == label_target_domain.data)
            total_for_domain_accuracy += label_source_domain.size(0)
            total_for_domain_accuracy += label_target_domain.size(0)
        
        # running_corrects_source = float(running_corrects_source)
        # running_corrects_target = float(running_corrects_target)
        # running_correct_domain = float(running_correct_domain)
        accuracy_total = (running_corrects_source.item() + running_corrects_target.item()) / \
                         (total_for_accuracy_source + total_for_accuracy_target)
        print('Accuracy total %4f,'
              ' main loss classifier %4f,'
              ' source accuracy %4f'
              ' source classification loss %4f,'
              ' target accuracy %4f'
              ' target loss %4f'
              ' accuracy domain distinction %4f'
              ' loss domain distinction %4f,'
              %
              (accuracy_total,
               loss_main_sum / n_total,
               running_corrects_source.item() / total_for_accuracy_source,
               loss_src_class_sum / n_total,
               running_corrects_target.item() / total_for_accuracy_target,
               loss_target_class_sum / n_total,
               running_correct_domain.item() / total_for_domain_accuracy,
               loss_domain_sum / n_total
               ))

        'VALIDATION STEP'
        running_loss_validation = 0.
        running_corrects_validation = 0
        total_validation = 0
        n_total_val = 0
        model.eval()
        for validation_batch in target_validation_dataset:
            # get the inputs
            inputs, labels = validation_batch
            inputs, labels = inputs, labels
            # zero the parameter gradients
            optimizer_classifier.zero_grad()

            with torch.no_grad():
                # forward
                outputs = model(inputs)
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

        loss_val = running_loss_validation / n_total_val
        scheduler.step(loss_val)
        if loss_val < best_loss:
            print("New best validation loss: ", loss_val)
            best_loss = loss_val
            # Load the BN statistics for the target
            BN_weights = copy.deepcopy(list_dictionaries_BN_weights[1])
            model.load_state_dict(BN_weights, strict=False)
            best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                          'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
            patience = epoch + patience_increment

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, max_epochs, time.time() - epoch_start))

        if patience < epoch:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_state


def run_SCADANN_training_sessions(examples_datasets, labels_datasets, num_kernels, feature_vector_input_length,
                                  path_weights_to_save_to="Weights_TSD/SCADANN",
                                  path_weights_Adversarial_training="Weights_TSD/DANN",
                                  path_weights_Normal_training="Weights_TSD/TSD",
                                  number_of_cycle_for_first_training=40, number_of_cycles_rest_of_training=40,
                                  number_of_classes=22, 
                                  percentage_same_gesture_stable=0.75,
                                  learning_rate=0.002515):
    """
    examples_datasets
    labels_datasets
    num_neurons
    feature_vector_input_length
    path_weights_to_save_to: path to save SCADANN weights
    path_weights_Adversarial_training: path to load DANN weights
    path_weights_Normal_training: path to load standard training weights
    number_of_cycle_for_first_training
    number_of_cycles_rest_of_training
    number_of_classes
    percentage_same_gesture_stable
    learning_rate
    """
    participants_train, _, _ = load_dataloaders_training_sessions(
        examples_datasets, labels_datasets, batch_size=128,
        number_of_cycle_for_first_training=number_of_cycle_for_first_training,
        number_of_cycles_rest_of_training=number_of_cycles_rest_of_training, drop_last=False, get_validation_set=False,
        shuffle=False)
    print("participants_train = ", len(participants_train))
    for participant_i in range(len(participants_train)):
        for session_j in range(1, len(participants_train[participant_i])):
            model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                feature_vector_input_length=feature_vector_input_length)

            # Define Loss functions
            cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean')

            # Define Optimizer
            learning_rate = 0.001316
            print("Optimizer = ", model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

            # Define Scheduler
            precision = 1e-8

            model, optimizer, _, start_epoch = load_checkpoint(
                model=model, optimizer=optimizer, scheduler=None,
                filename=path_weights_Adversarial_training + "/participant_%d/best_state_%d.pt" % (participant_i,
                                                                                                   session_j))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
                                                            verbose=True, eps=precision)
            
            # Load STD model for the first session
            # Load DANN models for the others (previous sessions including the current one)
            models_array = []
            for j in range(0, session_j + 1):
                model_temp = TSD_Network(number_of_class=number_of_classes, num_neurons=num_kernels,
                                         feature_vector_input_length=feature_vector_input_length)
                if j == 0:
                    model_temp, _, _, _ = load_checkpoint(
                        model=model_temp, optimizer=None, scheduler=None,
                        filename=path_weights_Normal_training + "/participant_%d/best_state_%d.pt" % (participant_i,
                                                                                                      j))
                else:
                    model_temp, _, _, _ = load_checkpoint(
                        model=model_temp, optimizer=None, scheduler=None,
                        filename=path_weights_Adversarial_training + "/participant_%d/best_state_%d.pt" % (
                            participant_i,
                            j))
                models_array.append(model_temp)
            print("==== models_array = ", np.shape(models_array), " @ session ", session_j)
            
            
            train_dataloader_replay, validationloader_replay, train_dataloader_pseudo, validationloader_pseudo = \
                generate_dataloaders_for_SCADANN(dataloader_sessions=participants_train[participant_i],
                                                 models=models_array,
                                                 current_session=session_j, validation_set_ratio=0.2, batch_size=64,
                                                 percentage_same_gesture_stable=percentage_same_gesture_stable)
            best_state = SCADANN_BN_training(replay_dataset_train=train_dataloader_replay,
                                             target_validation_dataset=validationloader_pseudo,
                                             target_dataset=train_dataloader_pseudo, model=model,
                                             crossEntropyLoss=cross_entropy_loss_classes,
                                             optimizer_classifier=optimizer,
                                             scheduler=scheduler, patience_increment=10, max_epochs=500,
                                             domain_loss_weight=1e-1)
            if not os.path.exists(path_weights_to_save_to + "/participant_%d" % participant_i):
                os.makedirs(path_weights_to_save_to + "/participant_%d" % participant_i)
            print(os.listdir(path_weights_to_save_to))
            torch.save(best_state, f=path_weights_to_save_to +
                                     "/participant_%d/best_state_%d.pt" % (participant_i, session_j))

def test_network_SLADANN(examples_datasets_train, labels_datasets_train, num_neurons, feature_vector_input_length,
                         path_weights_SCADANN ="Weights_TSD/SCADANN",
                         path_weights_normal="Weights_TSD/TSD",
                         algo_name="SCADANN", cycle_test=None, 
                         number_of_classes=22, save_path = 'results_tsd'):
    _, _, participants_test = load_dataloaders_training_sessions(examples_datasets_train, labels_datasets_train,
                                                                 batch_size=512, cycle_for_test=cycle_test)

    model_outputs = []
    predictions = []
    ground_truths = []
    accuracies = []
    for participant_index, dataset_test in enumerate(participants_test):
        model_outputs_participant = []
        predictions_participant = []
        ground_truth_participant = []
        accuracies_participant = []
        model = TSD_Network(number_of_class=number_of_classes, num_neurons=num_neurons,
                            feature_vector_input_length=feature_vector_input_length)
        for session_index, training_session_test_data in enumerate(dataset_test):

            if session_index == 0:
                best_state = torch.load(
                    path_weights_normal + "/participant_%d/best_state_%d.pt" %
                    (participant_index, 0))
            else:
                best_state = torch.load(
                    path_weights_SCADANN + "/participant_%d/best_state_%d.pt" %
                    (participant_index, session_index))
            best_weights = best_state['state_dict']
            model.load_state_dict(best_weights)

            predictions_training_session = []
            ground_truth_training_sesssion = []
            model_outputs_session = []
            with torch.no_grad():
                model.eval()
                for inputs, labels in training_session_test_data:
                    inputs = inputs
                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    model_outputs_session.extend(torch.softmax(output, dim=1).cpu().numpy())
                    predictions_training_session.extend(predicted.cpu().numpy())
                    ground_truth_training_sesssion.extend(labels.numpy())
            print("Participant: ", participant_index, " Accuracy: ",
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
        accuracies_to_display.append(accuracies_from_participant)
    print(accuracies_to_display)
    print("OVERALL ACCURACY: " + str(np.mean(accuracies_to_display)))

    file_to_open = save_path + "/" + algo_name + ".txt"
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
