import math
import pywt
import sampen
import numpy as np
from scipy import stats, spatial, signal
from itertools import combinations


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    lowcut_normalized = lowcut / nyq
    highcut_normalized = highcut / nyq
    b, a = signal.butter(N=order, Wn=[lowcut_normalized, highcut_normalized], btype='band', output="ba")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def get_TD_features_set(vector, threshold=1.):
    features = [getMAV(vector), getZC(vector, threshold=threshold), getSSC(vector, threshold=threshold), getWL(vector)]
    return np.array(features, dtype=np.float32)

def get_dataset_with_features_set(dataset, features_set_function=get_TD_features_set):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        all_zero_channel = False
        if features_set_function is getTSD:
            for vector_electrode in example:
                # If only 0. the sensor was not recording correctly and we should ignore this example
                if np.sum(vector_electrode) == 0:
                    all_zero_channel = True
            if all_zero_channel is False:
                example_formatted = features_set_function(example)
        else:
            for vector_electrode in example:
                # If only 0. the sensor was not recording correctly and we should ignore this example
                if np.sum(vector_electrode) != 0:
                    example_formatted.append(features_set_function(vector_electrode))
                else:
                    all_zero_channel = True
        if all_zero_channel is False:
            dataset_to_return.append(np.array(example_formatted).transpose().flatten())
    return dataset_to_return


def get_TD_features_set(vector, threshold=1.):
    features = [getMAV(vector), getZC(vector, threshold=threshold), getSSC(vector, threshold=threshold), getWL(vector)]
    return np.array(features, dtype=np.float32)

def getTSD(all_channels_data_in_window):
    """
        %Implementation from Rami Khusaba adapted to our code
        % References
        % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
        %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
        % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
        %     Neural Networks, vol. 55, pp. 42-58, 2014.
    """
    # x should be a numpy array
    all_channels_data_in_window = np.swapaxes(np.array(all_channels_data_in_window), 1, 0)

    if len(all_channels_data_in_window.shape) == 1:
        all_channels_data_in_window = all_channels_data_in_window[:, np.newaxis]

    datasize = all_channels_data_in_window.shape[0]
    Nsignals = all_channels_data_in_window.shape[1]

    # prepare indices of each 2 channels combinations
    # NCC = Number of channels to combine
    NCC = 2
    Indx = np.array(list(combinations(range(Nsignals), NCC)))

    # allocate memory
    # define the number of features per channel
    NFPC = 7

    # Preallocate memory
    feat = np.zeros((Indx.shape[0] * NFPC + Nsignals * NFPC))

    # Step1.1: Extract between-channels features
    ebp = getTSDfeatures_for_one_representation(
        all_channels_data_in_window[:, Indx[:, 0]] - all_channels_data_in_window[:, Indx[:, 1]])
    efp = getTSDfeatures_for_one_representation(
        np.log(
            (all_channels_data_in_window[:, Indx[:, 0]] - all_channels_data_in_window[:, Indx[:, 1]]) ** 2 + np.spacing(
                1)) ** 2)
    # Step 1.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[range(Indx.shape[0] * NFPC)] = num / den

    # Step2.1: Extract within-channels features
    ebp = getTSDfeatures_for_one_representation(all_channels_data_in_window)
    efp = getTSDfeatures_for_one_representation(np.log((all_channels_data_in_window) ** 2 + np.spacing(1)) ** 2)
    # Step2.2: Correlation analysis
    num = np.multiply(efp, ebp)
    den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
    feat[np.max(range(Indx.shape[0] * NFPC)) + 1:] = num / den
    return feat


def getTSDfeat_rami(x, *slidingParams):
    """
       % Implementation from Rami Khusaba
       % Time-domain power spectral moments (TD-PSD)
       % Using Fourier relations between time domina and frequency domain to
       % extract power spectral moments directly from time domain.
       %
       % References
       % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
       %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
       % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
       %     Neural Networks, vol. 55, pp. 42-58, 2014.
    """
    # x should be a numpy array
    x = np.array(x)

    # Make sure you have the correct number of parameters passed
    if len(slidingParams) < 2:
        raise TypeError(
            'getTSDfeat expected winsize and wininc to be passed, got %d parameters instead' % len(slidingParams))
    if slidingParams:
        winsize = slidingParams[0]
        wininc = slidingParams[1]

    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    datasize = x.shape[0]
    Nsignals = x.shape[1]
    numwin = np.int(np.floor((datasize - winsize) / wininc) + 1)

    # prepare indices of each 2 channels combinations
    # NCC = Number of channels to combine
    NCC = 2
    Indx = np.array(list(combinations(range(Nsignals), NCC)))

    # allocate memory
    # define the number of features per channel
    NFPC = 7
    print(datasize)

    # Preallocate memory
    feat = np.zeros((numwin, Indx.shape[0] * NFPC + Nsignals * NFPC))

    # prepare windowing analysis parameters
    st = 0
    en = winsize

    for i in range(numwin):
        # define your current window
        curwin = x[st:en, :]

        # Step1.1: Extract between-channels features
        ebp = getTSDfeatures_for_one_representation(curwin[:, Indx[:, 0]] - curwin[:, Indx[:, 1]])
        efp = getTSDfeatures_for_one_representation(
            np.log((curwin[:, Indx[:, 0]] - curwin[:, Indx[:, 1]]) ** 2 + np.spacing(1)) ** 2)
        # Step 1.2: Correlation analysis
        num = np.multiply(efp, ebp)
        den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
        feat[i, range(Indx.shape[0] * NFPC)] = num / den

        # Step2.1: Extract within-channels features
        ebp = getTSDfeatures_for_one_representation(curwin)
        efp = getTSDfeatures_for_one_representation(np.log((curwin) ** 2 + np.spacing(1)) ** 2)
        # Step2.2: Correlation analysis
        num = np.multiply(efp, ebp)
        den = np.sqrt(np.multiply(efp, efp)) + np.sqrt(np.multiply(ebp, ebp))
        feat[i, np.max(range(Indx.shape[0] * NFPC)) + 1:] = num / den

        # progress to next window
        st = st + wininc
        en = en + wininc

    return feat


def getTSDfeatures_for_one_representation(vector):
    """
    % Implementation from Rami Khusaba
    % Time-domain power spectral moments (TD-PSD)
    % Using Fourier relations between time domina and frequency domain to
    % extract power spectral moments dircetly from time domain.
    %
    % Modifications
    % 17/11/2013  RK: Spectral moments first created.
    % 02/03/2014  AT: I added 1 to the function name to differentiate it from other versions from Rami
    % 01/02/2016  RK: Modifed this code intosomewhat deep structure
    %
    % References
    % [1] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
    %     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
    % [2] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
    %     Neural Networks, vol. 55, pp. 42-58, 2014.
    """

    # Get the size of the input signal
    samples, channels = vector.shape

    if channels > samples:
        vector = np.transpose(vector)
        samples, channels = channels, samples

    # Root squared zero order moment normalized
    m0 = np.sqrt(np.nansum(vector ** 2, axis=0))[:, np.newaxis]
    m0 = m0 ** .1 / .1

    # Prepare derivatives for higher order moments
    d1 = np.diff(np.concatenate([np.zeros((1, channels)), vector], axis=0), n=1, axis=0)
    d2 = np.diff(np.concatenate([np.zeros((1, channels)), d1], axis=0), n=1, axis=0)

    # Root squared 2nd and 4th order moments normalized
    m2 = (np.sqrt(np.nansum(d1 ** 2, axis=0)) / (samples - 1))[:, np.newaxis]
    m2 = m2 ** .1 / .1

    m4 = (np.sqrt(np.nansum(d2 ** 2, axis=0)) / (samples - 1))[:, np.newaxis]
    m4 = m4 ** .1 / .1

    # Sparseness
    sparsi = m0 / np.sqrt(np.abs(np.multiply((m0 - m2) ** 2, (m0 - m4) ** 2)))

    # Irregularity Factor
    IRF = m2 / np.sqrt(np.multiply(m0, m4))

    # Coefficient of Variation
    COV = (np.nanstd(vector, axis=0, ddof=1) / np.nanmean(vector, axis=0))[:, np.newaxis]

    # Teager-Kaiser energy operator
    TEA = np.nansum(d1 ** 2 - np.multiply(vector[0:samples, :], d2), axis=0)[:, np.newaxis]

    # All features together
    STDD = np.nanstd(m0, axis=0, ddof=1)[:, np.newaxis]

    if channels > 2:
        Feat = np.concatenate((m0 / STDD, (m0 - m2) / STDD, (m0 - m4) / STDD, sparsi, IRF, COV, TEA), axis=0)
    else:
        Feat = np.concatenate((m0, m0 - m2, m0 - m4, sparsi, IRF, COV, TEA), axis=0)

    Feat = np.log(np.abs(Feat)).flatten()

    return Feat


if __name__ == '__main__':
    print("Compare TSD implementations")
    tsd_data = np.array(
        [[0.234631339, 0.569690978, 0.28497832, 0.767334908, 0.000929953, 0.955792684, 0.34489891, 0.901606785,
          0.833662576, 0.293860399],
         [0.022407932, 0.110714418, 0.173787089, 0.044756897, 0.608926394, 0.134628998, 0.373141251, 0.481913916,
          0.360470935, 0.199844388],
         [0.579233891, 0.596110557, 0.806037005, 0.889725032, 0.652525182, 0.193154786, 0.746598481, 0.627327263,
          0.255035701, 0.200252276]])
    own_implementation = getTSD(tsd_data)
    Rami_implementation = getTSDfeat_rami(np.swapaxes(tsd_data, 1, 0), *[len(tsd_data[0]), 1])[0]
    matlab_result = [1.663716179, 1.663840215, 1.728739776, 0.573757313, 0.580660839, 0.776983895, 0.345518206,
                     0.342801404, 0.767034427, 0.660166337, 0.659351174, 0.241340596, 0.060788158, 0.060782713,
                     0.063259494, 0.198698886, 0.340021142, 0.046130887, -0.034333537, 0.246710895, 0.011654321,
                     1.324898831, 1.280156697, 1.233062152, 0.348435412, 0.351258045, 0.301726381, 0.09836566,
                     0.14334697, 0.13194771, 0.531539406, 0.48664003, 0.350454092, 0.065954505, 0.069573256,
                     0.073204152, -0.310864682, -0.07100652, -0.209607288, -0.063863048, -1.070347941, -0.008215568]

    print(own_implementation)
    print(Rami_implementation)
    print(matlab_result)
    print("Own implementation vs Rami Python: ", np.allclose(own_implementation, Rami_implementation))
    print("Own implementation vs Rami Matlab: ", np.allclose(own_implementation, matlab_result))
