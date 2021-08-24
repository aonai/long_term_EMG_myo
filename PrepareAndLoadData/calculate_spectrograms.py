import numpy as np
from scipy import signal


def calculate_spectrogram_dataset(dataset, frequency):
    dataset_spectrogram = []
    for examples in dataset:
        canals = []
        for electrode_vector in examples:
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                calculate_spectrogram_vector(electrode_vector, fs=frequency, npserseg=100, noverlap=50)
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))

        example_to_classify = np.swapaxes(canals, 0, 1)
        dataset_spectrogram.append(example_to_classify)
    return dataset_spectrogram


def calculate_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window="hann",
                                                                                         scaling="density",
                                                                                         mode="magnitude")
    return spectrogram_of_vector, time_segment_sample, frequencies_samples


def calculate_single_canal_spectrogram(electrode_vector, frequency):
    # 150ms
    spectrogram_of_vector, time_segment_sample, frequencies_samples = \
        calculate_spectrogram_vector(electrode_vector, npserseg=20, noverlap=10, fs=frequency)
    # remove the low frequency signal as it's useless for sEMG (0-20Hz)
    spectrogram_of_vector = spectrogram_of_vector[1:]
    # frequencies_samples = frequencies_samples[1:]
    # print(time_segment_sample)
    # time_segment_sample = np.array(time_segment_sample.tolist().append(.150))
    # show_spectrogram(spectrogram_of_vector=spectrogram_of_vector, time_segment_sample=time_segment_sample,
    #                 frequencies_samples=frequencies_samples)
    return np.swapaxes(spectrogram_of_vector, 0, 1)


def calculate_single_example(example, frequency):
    canals = []
    for electrode_vector in example:
        canals.append(calculate_single_canal_spectrogram(electrode_vector, frequency))
    # show_input(canals)
    # print(np.shape(canals))
    return np.swapaxes(canals, 0, 1)
