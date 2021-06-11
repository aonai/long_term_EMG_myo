import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def segment_dataset_by_gesture_to_remove_transitions(ground_truths, predictions, model_outputs, examples):
    """
    Segment data based on where ground truth starts transit into another label. All data in type numpy
    
    Args:
        ground_truths: training dataset label
        predictions: output from model
        model_outputs: softmax of model (likelihood of each label)
        examples: training dataset
    
    Returns:
        ground_truth_segmented_session
        predictions_segmented_session
        model_outputs_segmented_session
        examples_segmented_session
    """
    ground_truth_segmented_session = []
    predictions_segmented_session = []
    model_outputs_segmented_session = []
    examples_segmented_session = []

    ground_truth_segmented_gesture = []
    predictions_segmented_gesture = []
    model_outputs_segmented_gesture = []
    examples_segmented_gesture = []
    current_label = ground_truths[0]
    for example_index in range(len(ground_truths)):
        if current_label != ground_truths[example_index]:
            ground_truth_segmented_session.append(ground_truth_segmented_gesture)
            predictions_segmented_session.append(predictions_segmented_gesture)
            model_outputs_segmented_session.append(model_outputs_segmented_gesture)
            examples_segmented_session.append(examples_segmented_gesture)
            ground_truth_segmented_gesture = []
            predictions_segmented_gesture = []
            model_outputs_segmented_gesture = []
            examples_segmented_gesture = []
            current_label = ground_truths[example_index]
        ground_truth_segmented_gesture.append(ground_truths[example_index])
        predictions_segmented_gesture.append(predictions[example_index])
        model_outputs_segmented_gesture.append(model_outputs[example_index])
        examples_segmented_gesture.append(examples[example_index])
    ground_truth_segmented_session.append(ground_truth_segmented_gesture)
    predictions_segmented_session.append(predictions_segmented_gesture)
    model_outputs_segmented_session.append(model_outputs_segmented_gesture)
    examples_segmented_session.append(examples_segmented_gesture)
    return ground_truth_segmented_session, predictions_segmented_session, model_outputs_segmented_session, \
           examples_segmented_session

def pseudo_labels_heuristic(predictions, model_outputs, window_stable_mode_length=5,
                            percentage_same_gesture_now_stable=0.75, maximum_length_instability_same_gesture=10,
                            maximum_length_instability_gesture_transition=10, use_look_back=False,
                            len_look_back=2):
    """
    Generate pseudo labels for one session
    unstable == transition in labels
        start collecting data into model_output_in_unstable_mode when transition occurs
        if actual transition between gestures if model output percentage > percentage_same_gesture_now_stable
            in this case, go back to collecting stable data 
            
            if current class == old class (that is gesture remain the same, the transition was a glitch)
                if length of unstable data < maximum_length_instability_gesture_transition
                    pseudo labels = current class during this unstable time length
                else: ignore all unstable data 
            else: actual transition 
                if length of unstable data < maximum_length_instability_gesture_transition
                    pseudo labels = current class during this unstable time length
                
                
        else: bad classifier 
    if still in stable mode, add current label to pseudo label
    
    data is deleted if model output stays unstbale for too long (> maximum_length_instability_gesture_transition):
        and also unstable data won't be checked if output < percentage_same_gesture_now_stable (no definite output)
    
    Args:
        predictions: output from model
        model_outputs: softmax of model (likelihood of each label)
        window_stable_mode_length: length of inputs to check whether it is stable
        percentage_same_gesture_now_stable: accuracy of a stable array of input should 
        maximum_length_instability_same_gesture: maximum length of inputs that can stay stable
        maximum_length_instability_gesture_transition: maximum length of inputs that can stay unstable
        use_look_back: used only for eval part (check this later)
        len_look_back
    
    Returns:
        predictions_heuristic: array of pseudo labels 
        predictions_heuristic_index: array of corresponding index of pseudo labels in the original labels 
    """
    
    
    predictions_heuristic = []
    predictions_heuristic_index = []
    # current class = most labels appeared in the first <window_stable_mode_length> portion of model_outputs
    current_class = np.argmax(np.median(model_outputs[0:window_stable_mode_length], axis=0))
    # print("=== current class = ", np.median(model_outputs[0:window_stable_mode_length], axis=0))
    # print("  === predicted ", predictions)
    stable_predictions = True

    index_unstable_start = 0
    model_output_in_unstable_mode = []
    for index, (prediction, model_output) in enumerate(zip(predictions,
                                                           model_outputs)):
        if prediction != current_class and stable_predictions:
            stable_predictions = False
            index_unstable_start = index

            model_output_in_unstable_mode = []

        if stable_predictions is False:
            model_output_in_unstable_mode.append(model_output)
        if len(model_output_in_unstable_mode) > window_stable_mode_length:
            model_output_in_unstable_mode = model_output_in_unstable_mode[1:]

        if stable_predictions is False:
            if len(model_output_in_unstable_mode) >= window_stable_mode_length:
                '''
                mode_gesture_frequency = mode(model_output_in_unstable_mode)[1][0]
                most_prevalent_gesture = mode(model_output_in_unstable_mode)[0][0]

                print(mode_gesture)
                '''
                # print("=== model_output_in_unstable_mode ", np.shape(model_output_in_unstable_mode))
                medians = np.median(np.array(model_output_in_unstable_mode), axis=0)
                medians_percentage = medians / np.sum(medians)
                most_prevalent_gesture = np.argmax(medians_percentage)
                # print("   === most_prevalent_gesture ", most_prevalent_gesture)
                # print("   === medians_percentage ", medians_percentage)
                # print(" === checking on percentage tresh ", medians_percentage[most_prevalent_gesture], " @ index ", index)
                if medians_percentage[most_prevalent_gesture] > percentage_same_gesture_now_stable:
                    stable_predictions = True
                    # Determine if this period of instability was due to a gesture transition or if it was due to
                    # a bad classification from the classifier
                    old_class = current_class
                    current_class = most_prevalent_gesture
                    # After the unstability, we are still on the same gesture. Check if it took too long to solve
                    # If it was ignore all the predictions up to that point. If the period of instability did not
                    # last too long, correct all predictions to be of the current class
                    if current_class == old_class:
                        if index - index_unstable_start < maximum_length_instability_same_gesture:
                            # Add all the predictions made during instability to the pseudo labels and set them to
                            # the current class to the current class
                            for index_to_add in range(index_unstable_start, index + 1):
                                predictions_heuristic.append(current_class)
                                predictions_heuristic_index.append(index_to_add)
                        # Else: Ignore all the predictions made during the period of instability.
                    else:  # current class != old_class, therefore we just experienced a transition. Act accordingly
                        if index - index_unstable_start < maximum_length_instability_gesture_transition:
                            # Add all the predictions made during instability to the pseudo labels and set them to
                            # to the new class as they were all part of a transition between two gestures
                            for index_to_add in range(index_unstable_start, index + 1):
                                predictions_heuristic.append(current_class)
                                predictions_heuristic_index.append(index_to_add)
                            """ This part is for eval
                            if use_look_back:
                                index_from_where_to_change = look_back_and_re_label(
                                    network_outputs=model_outputs,
                                    index_start_look_back=index_unstable_start,
                                    len_look_back=len_look_back)

                                index_from_where_to_change_in_pseudolabels = None
                                for i in range(index_from_where_to_change,
                                               index_unstable_start):
                                    if i in predictions_heuristic_index:
                                        index_from_where_to_change_in_pseudolabels = \
                                            predictions_heuristic_index.index(i)
                                        break
                                # print(predictions_heuristic)
                                if index_from_where_to_change_in_pseudolabels is not None:
                                    for index_to_change in range(index_from_where_to_change_in_pseudolabels,
                                                                 np.min((
                                                                         index_from_where_to_change_in_pseudolabels +
                                                                         len_look_back,
                                                                         len(predictions_heuristic)))):
                                        predictions_heuristic[index_to_change] = current_class
                                # print(predictions_heuristic)
                            """ 
                        # Else: Ignore all the predictions made during the period of instability.
                    model_output_in_unstable_mode = []
        else:
            predictions_heuristic.append(prediction)
            predictions_heuristic_index.append(index)

    # Add the remaining few predictions to the current gestures (in actual real-time scenario, that last part would
    # not exist. But we are working with a finite dataset
    if stable_predictions is False and \
            len(predictions) - index_unstable_start < window_stable_mode_length * percentage_same_gesture_now_stable:
        gesture_to_add = np.argmax(np.median(model_outputs, axis=0))
        for index_to_add in range(index_unstable_start, len(predictions)):
            predictions_heuristic.append(gesture_to_add)
            predictions_heuristic_index.append(index_to_add)

    return predictions_heuristic, predictions_heuristic_index

def pseudo_labels_heuristic_training_sessions(predictions, model_outputs, window_stable_mode_length=5,
                                              percentage_same_gesture_now_stable=0.75,
                                              maximum_length_instability_same_gesture=10,
                                              maximum_length_instability_gesture_transition=10, 
                                              use_look_back=False,
                                              len_look_back=2):
    """
    wrapper for generate pseudo labels for each session
    
    Args:
        predictions: output from model
        model_outputs: softmax of model (likelihood of each label)
        window_stable_mode_lengh: length of inputs to check whether it is stable
        percentage_same_gesture_now_stable: accuracy of a stable array of input should 
        maximum_length_instability_same_gesture: maximum length of inputs that can stay stable
        maximum_length_instability_gesture_transition: maximum length of inputs that can stay unstable
        use_look_back: used only for eval part (check this later)
        len_look_back
    
    Returns: 
        pseudo_labels_sessions: ndarray of pseudo labels for each session
        indexes_associated_with_pseudo_labels_sessions: ndarray of corresponding index of pseudo labels 
                                                        for each session
    """
    pseudo_labels_sessions = []
    indexes_associated_with_pseudo_labels_sessions = []
    for session_index in range(len(predictions)):
        pseudo_labels, indexes_associated = pseudo_labels_heuristic(
            predictions[session_index], model_outputs[session_index],
            window_stable_mode_length=window_stable_mode_length,
            percentage_same_gesture_now_stable=percentage_same_gesture_now_stable,
            maximum_length_instability_same_gesture=maximum_length_instability_same_gesture,
            maximum_length_instability_gesture_transition=maximum_length_instability_gesture_transition,
            use_look_back=use_look_back,
            len_look_back=len_look_back)
        pseudo_labels_sessions.append(pseudo_labels)
        indexes_associated_with_pseudo_labels_sessions.append(indexes_associated)
    return pseudo_labels_sessions, indexes_associated_with_pseudo_labels_sessions

def generate_dataloaders_for_SCADANN(dataloader_sessions, models, current_session, validation_set_ratio=0.2,
                                     batch_size=128, percentage_same_gesture_stable=0.75):
    """
    Wrapper for generating pseudo labeled dataloaders for one participant and one session
    Use dataset labels for first session, pseudo_labels for the others (seconds to current sessions)
    
    Args:
        dataloader_sessions: dataloader for one participant 
        models: models for one participant
        current_session: current training session
        validation_set_ratio: percentage of validation set
        batch_size: size of one batch in dataloader
        percentage_same_gesture_stable: accuracy of a stable array of input should 
        
    Returns:
        train_dataloader_replay: train data of previous sessions (including the first)
        validationloader_replay: validation data of previous sessions (including the first)
        train_dataloader_pseudo: train data of current session
        validationloader_pseudo: validation data of current session
    """
    examples_replay = []
    labels_replay = []

    examples_new_session = []
    labels_new_session = []
    for session_index in range(current_session + 1):
        print("HANDLING NEW SESSION ", session_index)
        # This is the first session where we have real labels. Use them
        models[session_index].eval()
        if session_index == 0:
            # Create the training and validation dataset to train on
            for batch in dataloader_sessions[0]:
                with torch.no_grad():
                    inputs, labels_batch = batch
                    labels_replay.extend(labels_batch.numpy())
                    examples_replay.extend(inputs.numpy())
        # We don't have true labels for these sessions, generated them
        # elif session_index == current_session:
        else:
            examples_self_learning = []
            ground_truth_self_learning = []
            model_outputs_self_learning = []
            predicted_labels_self_learning = []
            # Get the predictions and model output for the pseudo label dataset
            models[session_index].eval()
            # model.apply(model.apply_dropout)
            for batch in dataloader_sessions[session_index]:
                with torch.no_grad():
                    inputs_batch, labels_batch = batch
                    inputs_batch = inputs_batch

                    outputs = models[session_index](inputs_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    model_outputs_self_learning.extend(torch.softmax(outputs, dim=1).cpu().numpy())

                    # print("=== predicted ", np.shape(predicted.numpy()))
                    # print("  === examples ", np.shape(inputs_batch.numpy()))
                    # print("  === ground_truth ", np.shape(labels_batch.numpy()))
                    # print("  === model_outputs_self_learning : ", np.shape(torch.softmax(outputs, dim=1).cpu().numpy()))
                    predicted_labels_self_learning.extend(predicted.cpu().numpy())
                    ground_truth_self_learning.extend(labels_batch.numpy())
                    examples_self_learning.extend(inputs_batch.cpu().numpy())
            # Segment the data in respect to the recorded gestures
            ground_truth_segmented, predictions_segmented, model_outputs_segmented, examples_segmented = \
                segment_dataset_by_gesture_to_remove_transitions(ground_truths=ground_truth_self_learning,
                                                                 predictions=predicted_labels_self_learning,
                                                                 model_outputs=model_outputs_self_learning,
                                                                 examples=examples_self_learning)
            print("Finish segment dataset")

            pseudo_labels_segmented, indexes_pseudo_labels_segmented = pseudo_labels_heuristic_training_sessions(
                predictions_segmented, model_outputs_segmented, window_stable_mode_length=10,
                percentage_same_gesture_now_stable=percentage_same_gesture_stable,
                maximum_length_instability_gesture_transition=15,
                maximum_length_instability_same_gesture=15)
            print("Finish pseudo_labels")
            
            # Put all pseudo labels together (with unstable portion removed)
            pseudo_labels = []
            examples = []
            ground_truth_reduced = []
            for index_segment in range(len(pseudo_labels_segmented)):
                pseudo_labels.extend(pseudo_labels_segmented[index_segment])
                print("BEFORE: ", np.mean(np.array(ground_truth_segmented[index_segment]) ==
                                          np.array(predictions_segmented[index_segment])), "  AFTER: ", np.mean(
                    np.array(ground_truth_segmented[index_segment])[indexes_pseudo_labels_segmented[index_segment]] ==
                    np.array(pseudo_labels_segmented[index_segment])), " len before: ",
                      len(ground_truth_segmented[index_segment]), "  len after: ",
                      len(pseudo_labels_segmented[index_segment]))
                examples.extend(np.array(examples_segmented[index_segment])[
                                    indexes_pseudo_labels_segmented[index_segment]].tolist())
                ground_truth_reduced.extend(np.array(ground_truth_segmented[index_segment])[
                                                indexes_pseudo_labels_segmented[index_segment]].tolist())

            accuracy_before = np.mean(np.array(ground_truth_self_learning) ==
                                      np.array(predicted_labels_self_learning))
            accuracy_after = np.mean(np.array(ground_truth_reduced) ==
                                     np.array(pseudo_labels))
            print("ACCURACY MODEL: ", accuracy_before, "  Accuracy pseudo:", accuracy_after, " len pseudo: ",
                  len(pseudo_labels), "   len predictions", len(predicted_labels_self_learning))

            if session_index == current_session:
                examples_new_session.extend(examples)
                labels_new_session.extend(pseudo_labels)
            else:
                examples_replay.extend(examples)
                labels_replay.extend(pseudo_labels)

    # Create the dataloaders associated with the REPLAY data
    X_replay, X_valid_replay, Y_replay, Y_valid_replay = train_test_split(examples_replay, labels_replay,
                                                                          test_size=validation_set_ratio, shuffle=True)

    validation_replay = TensorDataset(torch.from_numpy(np.array(X_valid_replay, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_replay, dtype=np.int64)))
    validationloader_replay = torch.utils.data.DataLoader(validation_replay, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_replay = TensorDataset(torch.from_numpy(np.array(X_replay, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_replay, dtype=np.int64)))
    train_dataloader_replay = torch.utils.data.DataLoader(train_replay, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    # Create the dataloaders associated with the NEW session
    X_pseudo, X_valid_pseudo, Y_pseudo, Y_valid_pseudo = train_test_split(examples_new_session, labels_new_session,
                                                                          test_size=validation_set_ratio, shuffle=True)

    validation_pseudo = TensorDataset(torch.from_numpy(np.array(X_valid_pseudo, dtype=np.float32)),
                                      torch.from_numpy(np.array(Y_valid_pseudo, dtype=np.int64)))
    validationloader_pseudo = torch.utils.data.DataLoader(validation_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=False)
    train_pseudo = TensorDataset(torch.from_numpy(np.array(X_pseudo, dtype=np.float32)),
                                 torch.from_numpy(np.array(Y_pseudo, dtype=np.int64)))
    train_dataloader_pseudo = torch.utils.data.DataLoader(train_pseudo, batch_size=batch_size, shuffle=True,
                                                          drop_last=True)

    return train_dataloader_replay, validationloader_replay, train_dataloader_pseudo, validationloader_pseudo
