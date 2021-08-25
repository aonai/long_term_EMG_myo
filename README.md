# Long-term EMG Myo Library
* Sonia Yuxiao Lai
* MSR Final Project  
&nbsp;
* Original by [Ulysse Côté-Allard](https://github.com/UlysseCoteAllard/LongTermEMG)
* Dataset recorded by [Suguru Kanoga](https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset)

### Overview
The goal of this project is to determine whether [Unsupervised Transfer Learning proposed by Ulysse Côté-Allard](https://ieeexplore.ieee.org/document/9207910) is feasible for a low-end wearable sensor, such as Myo armband, which has 8 channels sampled at 200Hz. This repository provides a library adapted for training [Myo sEMG signals](https://www.sciencedirect.com/science/article/pii/S1746809420301373) recorded from 5 subjects over 40-42 days with three different wearing locations. Users can choose to train and test models across-subject, days, and wearing locations.


### Usage 
Refer to [long-term myo notes](https://github.com/aonai/long_term_myo_notes) for complete code

##### 0. Prepare Data
i. Specify the following file locations:
* dataset: `data_dir`
* processed dataset: `processed_data_dir`
* trained model weights: `path_base_weights`, `path_DANN_weights`, `path_SCADANN_weights`
* model test results: `path_result` 

ii. Process raw datasets into 
* formatted example arrays with 7 features (proposed by [Rami N. Khushaba](https://github.com/RamiKhushaba/getTSDfeat))
`read_data_training(path=data_dir, store_path=processed_data_dir)`
* or formatted spectrograms of shape (4, 8, 10) 
`read_data_training(path=data_dir, store_path=processed_data_dir, spectrogram=True)`   

iii.Load processed examples and labels   
```
with open(processed_data_dir + "/training_session.pickle", 'rb') as f:
    dataset_training = pickle.load(file=f)
examples_datasets_train = dataset_trainin['examples_training']
labels_datasets_train = dataset_training['labels_training']
```  

iv. Specify the following training parameters: 
* `num_kernels`: list of integers where each integer corresponds to number of kernels of each layer for the base model
* `filter_size`:  a 2d list of shape (m, 2), where m is number of levels and each list corresponds to kernel size of the base ConvNet model
* `number_of_classes`: total number of gestures recorded
* `number_of_cycles_total`: total number of trails in one session
* `feature_vector_input_length`: length of one formatted example (252) for the base TSD model
* `batch_size` and `learning_rate` 


##### 1. Base Model  
i. Train a base model using Temporal-Spatial Descriptors (TSD) with `neural_net='TSD'` or Convolutional Network (ConvNet) with `neural_net='ConvNet'`  
```
train_fine_tuning(examples_datasets_train, labels_datasets_train,
            num_kernels=<num_kernels_of_each_layer>,   
            path_weight_to_save_to=<path_base_weights>,  
            number_of_classes=<number_of_classes>,   
            number_of_cycles_total=<number_of_cycles_total>,
            batch_size=<batch_size>,  
            feature_vector_input_length=<length_of_formatted_example>,
            learning_rate=<learning_rate>,  
            neural_net=<choise_of_model>,
            filter_size=<kernel_size_of_ConvNet>)
```
ii. Test and record results of the base model
```
test_standard_model_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                num_neurons=<num_kernels_of_each_layer>,  
                                use_only_first_training=<whether_to_use_fine_tuned model>,
                                path_weights=<path_base_weights>,
                                save_path=<path_result>,   
                                algo_name=<result_file_name>,
                                number_of_cycles_total=<number_of_cycles_total>,  
                                number_of_classes=<number_of_classes>,  
                                cycle_for_test=<testing_session_num>,
                                neural_net=<choise_of_model>,
                                filter_size=<kernel_size_of_ConvNet>)
```                             
##### 2. Domain-Adversarial Neural Network (DANN)  
i. Train a DANN model from the base model
```
train_DANN(examples_datasets_train, labels_datasets_train, 
        num_kernels=<num_kernels_of_each_layer>,
        path_weights_fine_tuning=<path_base_weights>,
        number_of_classes=<number_of_classes>,
        number_of_cycles_total=<number_of_cycles_total>,
        batch_size=<batch_size>,
        path_weights_to_save_to=<path_DANN_weights>, 
        learning_rate=<learning_rate>,
        neural_net=<choise_of_model>,
        filter_size=<kernel_size_of_ConvNet>)
```
ii. Test and record results of DANN model
```
test_DANN_on_training_sessions(examples_datasets_train, labels_datasets_train,
                            num_neurons=<num_kernels_of_each_layer>,  
                            path_weights_DA=<path_DANN_weights>,
                            algo_name=<result_file_name>, 
                            save_path=<path_result>, 
                            number_of_cycles_total=<number_of_cycles_total>,
                            path_weights_normal=<path_base_weights>, 
                            number_of_classes=<number_of_classes>,
                            cycle_for_test=<testing_session_num>, 
                            neural_net=<choise_of_model>,
                            filter_size=<kernel_size_of_ConvNet>)
```

##### 3. Self-Calibrating Asynchronous Domain Adversarial Neural Network (SCADANN) 
i. Train a Self-Calibrating Asynchronous Domain Adversarial Neural Network (SCADANN) model from the DANN model and base model
```
run_SCADANN_training_sessions(examples_datasets=examples_datasets_train, labels_datasets=labels_datasets_train,  
                        num_kernels=<num_kernels_of_each_layer>, 
                        path_weights_to_save_to=<path_SCADANN_weights>,
                        path_weights_Adversarial_training=<path_DANN_weights>,
                        path_weights_Normal_training=<path_base_weights>,
                        number_of_cycles_total=<number_of_cycles_total>, 
                        number_of_classes=<number_of_classes>,
                        learning_rate=<learning_rate>, 
                        neural_net=<choise_of_model>,
                        filter_size=<kernel_size_of_ConvNet>)
```
ii. Test and record results of SCADANN model
```
test_network_SLADANN(examples_datasets_train=examples_datasets_train, 
                    labels_datasets_train=labels_datasets_train,
                    num_neurons=<num_kernels_of_each_layer>, 
                    path_weights_SCADANN =<path_SCADANN_weights>, 
                    path_weights_normal=<path_base_weights>,
                    algo_name=<result_file_name>, 
                    cycle_test=<testing_session_num>,
                    number_of_cycles_total=<number_of_cycles_total>,
                    number_of_classes=<number_of_classes>,  
                    save_path = <path_result>, 
                    neural_net=<choise_of_model>,
                    filter_size=<kernel_size_of_ConvNet>)
```
##### 4. Load Results
i. Load results from stored npy file
``` 
results_filename = <path_result> + '/predictions_' + <result_file_name> + '.npy'
results = np.load(results_filename, allow_pickle=True)
accuracies = results[0]
``` 
Each npy result file includes accuracies, predictions, ground truths, and model outputs in order.

ii. Generate accuracies for each gestures 
```
ground_truths = results[1]
predictions = results[2]
df = get_gesture_accuracies(ground_truths, predictions,
                            number_of_classes=<number_of_classes>, 
                            m_name=<name_of_first_controlling_factor>,
                            n_name=<name_of_second_controlling_factor,
                            path=<path_result>, algo_name=<result_file_name>)
df = pd.read_csv(<path_result>+'/'+<result_file_name>+'.csv')
```


### Results
In conclusion, SCADANN is feasible in improving training performance for sEMG signals recorded by low-end wearable sensors. Compared to the original TSD, SCADANN improves accuracy by 11±4.6% (avg±sd), 9.6±5.0%, and 9.3±3.5% across all possible user-to-user, day-to-day, and location-to-location cases, respectively.
The amount of improvement from the base model to SCADANN is random. In a best-case scenario, accuracy increases by 28% (from 28% to 56%). Good growth does not guarantee a good model. In a similar case, accuracy increases by 9% (from 56% to 65%) while obtaining a good resulting accuracy. The performance of SCADANN overall conditions is dependant on how good the base model is while being used in other situations. For instance, when testing models on subject 0 at neutral location, TSD accuracy trained using subject 1 is 19% higher (39% compared to 58%) and SCADANN accuracy is 14% higher (50% compared to 64%) than trained using subject 3. To make the best performance of this unsupervised transfer learning, the base model needs to be of good accuracy as well. This argument is proved by comparing the performance of ConvNet and TSD as the base model. Compared to ConvNet, TSD is better at classifying Myo gestures; it improves accuracy from 77% to 81%, whereas ConvNet improves from 63% to 69% when both models are trained using the same controlling factors. More detailed results can be found in [long-term myo notes](https://github.com/aonai/long_term_myo_notes).