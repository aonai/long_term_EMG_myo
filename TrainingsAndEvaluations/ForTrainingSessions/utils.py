import numpy as np
import pandas as pd

def get_gesture_accuracies(ground_truths, predictions, number_of_classes=22, m_name="Sub", n_name="Loc", 
                        path='results', algo_name="gesture_accuracies", start_at_participant=1, num_participant=5,
                        index_participant_list_customized=None, lump_day_at_participant=None):
    """
    helper function to extract accuracies for each gesture on each condition

    Args:
        ground_truths: ndarray/list of all training ground truth in shape (m, n)
        predictions: ndarray/list of all predictions in shape (m, n)
        number_of_classes: number of gestures to extract accuracies for
        m_name: name of first comparison case 
        n_name: name of second comparison case 
            ex. if comparing inter-location training results of myo data set, m =  5 and n = 3.
                In this case, m_name should be "Sub" and n_name should be "Loc".
                As a result gesture accuracies for Participant 0 and Wearing Location 0 is a list
                of 22 accuracies under column name "Sub0_Loc0"
        path: where to save resulting csv file
        algo_name: nickname of model (this will be included in file name of results)
        start_at_participant: parameter for recording inter-subject training results num_participant
        num_participant: numer of participants to include; should be integer between 1~5
        index_participant_list_customized: customzied column names for lumped training
        lump_day_at_participant: param for lumped day training, specify which subject is included 

    Returns:
        accuracies_gestures: ndarray that stores accuracies for each gesture
        Also stores a csv file at given file location
    """
    column_names = []
    accuracies_gestures = [ [] for _ in range(number_of_classes) ]
    if index_participant_list_customized:
        index_participant_list = index_participant_list_customized
    else:
        if start_at_participant == 1:
            index_participant_list = list(range(num_participant))
        else:
            index_participant_list = list(range(start_at_participant-1,num_participant))   
            if len(index_participant_list) < num_participant:
                index_participant_list.extend(list(range(start_at_participant-1))) 
    print("index_participant_list ", index_participant_list)
    
    for m, ground_list in enumerate(ground_truths):
        for n, ground in enumerate(ground_list):
            if 'Sub' in n_name or lump_day_at_participant:
                column_names.append(f"{m_name}{lump_day_at_participant}_{n_name}{index_participant_list[n]}")
            else:
                column_names.append(f"{m_name}{m}_{n_name}{n+start_at_participant-1}")
            
            pred = predictions[m][n]
            #print("ground  = ", np.shape(ground))
            #print("pred = ", np.shape(pred))
            
            pred_tmp = [ [] for _ in range(number_of_classes) ]
            for i, g in enumerate(ground):
                p = pred[i]
                pred_tmp[g].append(p)

            # print("gestures pred = ", np.shape(pred_tmp))
            
            for gesture_idx, gesture_pred in enumerate(pred_tmp):
                acc = np.mean(np.array(gesture_pred) == np.ones(np.shape(gesture_pred))*gesture_idx)
                accuracies_gestures[gesture_idx].append(acc)
    print("accuracies_gestures = ", np.shape(accuracies_gestures))

    index_names = [f'M{i}' for i in range(number_of_classes)]
    accuracies_gestures_df  = pd.DataFrame(accuracies_gestures,
                                            index = index_names,
                                            columns = column_names)
    accuracies_gestures_df.loc['Mean'] = accuracies_gestures_df.mean()
    accuracies_gestures_df.to_csv(path+'/'+algo_name+'.csv')
    return accuracies_gestures_df