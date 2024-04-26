# script to extract 448 univariate time and frequency domain features from processed EEGs
# Based on Leal, Adriana, et al. "Unsupervised EEG preictal interval identification in patients with drug-resistant epilepsy." 
# Scientific Reports 13.1 (2023): 784.

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis 
from scipy.integrate import simpson
import statsmodels.api as sm
from eeglib.helpers import Helper
from eeglib.eeg import EEG
from joblib import Parallel, delayed
import sys

def statistical_measures(eeg):
    """
    Input:
    eeg - numpy array (channels as rows and time as column)
    
    Output:
    Dict with vectors of mean, standard deviation, skewness, kurtosis 
    For mapping values in vectors to channels: vector index = row index of eeg         
    """
    return {"mean": np.mean(eeg, 1),
            "std": np.std(eeg, 1),
            "skewness": skew(eeg, 1),
            "kurtosis": kurtosis(eeg,1)}


def decorrelation_time(eeg):
    """
    Input:
    eeg - numpy array (channels as rows and time as column)
    
    Output:
    vector decorrelation times
    For mapping values in the output vector to channels: vector index = row index of eeg 
    """
    # if no decorr time found within 200 then output it as nan
    decorr = np.empty((eeg.shape[0],))
    for i in range(eeg.shape[0]):
        # compute autocorrelation
        # the default value for nlags = 10*np.log10(length of vector) in acf()
        # for our case of 2000 elements in vector, default nlags ~ 35
        # Here, calculating upto 1000 lags
        acf = sm.tsa.acf(eeg[i,:], nlags=1000)
        decorr[i] = np.argmax(acf<=0)
        # if no decorr time found within 1000, i.e. output of np.argmax() is 0
        # then output decorr as 10000 (just some very large number)
        if decorr[i] == 0:
            decorr[i] = 10000
    
    return decorr.astype(int)


def df_to_eeg_helper(df, columns, window_size, high_pass=50.0, low_pass=1.0, normalize=False):
    """
    Input:
    df - Dataframe containing eeg signals (row: time points, columns: EEG channels) 
    columns - Name of EEG channels corresponding to the columns of df
    window_size - Signal window size
    high_pass, low_pass - Frequency values to apply low/high bandpass filters
    normalize - whether to normalize tthe signal or not
    
    Output:
    class object of eeg.lib 
    """
    data = df[columns].copy().to_numpy().transpose()
    helper = Helper(
        data, 
        sampleRate=200, 
        names=columns, 
        windowSize=window_size,
        highpass=high_pass,
        lowpass=low_pass,
        normalize=normalize
    )
    return helper

## Adapted from https://github.com/adrianaleal/eeg-preictal-identification-epilepsy/blob/main/1_extract_eeg_features/FunctionsFeatures/LinearUnivariate/spectral_edge.m
def spectral_edge_frequency(frequency, psd):
    """
    Input:
    frequency - vector with frequency at which PSD was computed by EEG.PSD()
    psd - vector with PSD values from EEG.PSD()
    
    Output:
    sef - Spectral edge freqeuncy
    """

    # get the indexes of the 0–40 Hz frequency band
    indices = np.argwhere(frequency<=40)

    # get the 0–40 Hz frequency and power vectors
    frequency_band = frequency[indices]
    power_band = psd[indices]

    # get the total power in that band
    total_spectral_power_band = sum(power_band)

    # define the percentage of overall power
    power_threshold = 50

    # get the the corresponding value of power
    spectral_power_threshold = total_spectral_power_band*(power_threshold/100)

    # get the cumulative power
    cumulative_power_band = np.cumsum(power_band)

    # get the the corresponding value of frequency and power
    ind_spectral_edge = np.argwhere(cumulative_power_band>=spectral_power_threshold)[0]
    spectral_edge_frequency = frequency_band[ind_spectral_edge]

    return spectral_edge_frequency[0]


def eeglib_features(helper):
    """
    Input:
    helper class object of eeglib package
    
    Output:
    Univariate frequency domain and some time domain features extracted using eeglib functions   
    """
    # get eeg stored as part of helper object
    eeg = [eeg for eeg in helper][0]

    # Absolute Band Power values in db
    # spectrumFrom='PSD' for consistency with how spectrum edge frequency is obtained from PSD obtained by similar method
    bp = eeg.bandPower(bands={'alpha': (8, 12), 'beta': (13, 20), 'delta': (1, 4), 'theta': (4, 7)},
                       spectrumFrom='PSD')
    # Split band power 
    bp_alpha = [ch["alpha"] for ch in bp]
    bp_beta = [ch["beta"] for ch in bp]
    bp_delta = [ch["delta"] for ch in bp]
    bp_theta = [ch["theta"] for ch in bp]

    # Power Spectral Density
    freq_psd = eeg.PSD(retFrequencies=True)

    #Higuchi fractal dimension
    hfd = list(eeg.HFD())
    # Hjorth parameters
    hjorth_mobility = list(eeg.hjorthMobility())
    hjorth_complexity = list(eeg.hjorthComplexity())
    # detrended fluctuation analysis
    dfa = list(eeg.DFA())
    # sample entropy
    sampEn = list(eeg.sampEn())

    df_as_dict = {}

    # separate out the feature values for each of the channels in eeg
    for chan_idx, col_name in enumerate(helper.names):
        df_as_dict[f"{col_name}.abs_bp_alpha"] = [bp_alpha[chan_idx]]
        df_as_dict[f"{col_name}.abs_bp_beta"] = [bp_beta[chan_idx]]
        df_as_dict[f"{col_name}.abs_bp_delta"] = [bp_delta[chan_idx]]
        df_as_dict[f"{col_name}.abs_bp_theta"] = [bp_theta[chan_idx]]

         # Total Power 
        df_as_dict[f"{col_name}.total_power"] = simpson(y = freq_psd[chan_idx,1,:], x = freq_psd[chan_idx,0,:])

        # Relative band power
        df_as_dict[f"{col_name}.rel_bp_alpha"] = [bp_alpha[chan_idx]]/df_as_dict[f"{col_name}.total_power"]
        df_as_dict[f"{col_name}.rel_bp_beta"] = [bp_beta[chan_idx]]/df_as_dict[f"{col_name}.total_power"]
        df_as_dict[f"{col_name}.rel_bp_delta"] = [bp_delta[chan_idx]]/df_as_dict[f"{col_name}.total_power"]
        df_as_dict[f"{col_name}.rel_bp_theta"] = [bp_theta[chan_idx]]/df_as_dict[f"{col_name}.total_power"]
        
        #[delta/alpha, delta/beta, delta/theta, theta/alpha, theta/beta, alpha/beta, beta/(alpha+theta), and theta/(alpha+beta)
        df_as_dict[f"{col_name}.bp_delta_alpha"] = [bp_delta[chan_idx]/bp_alpha[chan_idx]]
        df_as_dict[f"{col_name}.bp_delta_beta"] = [bp_delta[chan_idx]/bp_beta[chan_idx]]
        df_as_dict[f"{col_name}.bp_delta_theta"] = [bp_delta[chan_idx]/bp_theta[chan_idx]]
        df_as_dict[f"{col_name}.bp_theta_alpha"] = [bp_theta[chan_idx]/bp_alpha[chan_idx]]
        df_as_dict[f"{col_name}.bp_theta_beta"] = [bp_theta[chan_idx]/bp_beta[chan_idx]]
        df_as_dict[f"{col_name}.bp_alpha_beta"] = [bp_alpha[chan_idx]/bp_beta[chan_idx]]
        df_as_dict[f"{col_name}.bp_beta_alpha+theta"] = [bp_beta[chan_idx]/(bp_alpha[chan_idx]+bp_theta[chan_idx])]
        df_as_dict[f"{col_name}.bp_theta_alpha+beta"] = [bp_theta[chan_idx]/(bp_alpha[chan_idx]+bp_beta[chan_idx])]

        # Spectral edge frequency
        df_as_dict[f"{col_name}.spectral_edge_freq"] = spectral_edge_frequency(freq_psd[chan_idx,0,:], freq_psd[chan_idx,1,:])

        df_as_dict[f"{col_name}.hfd"] = [hfd[chan_idx]]
        df_as_dict[f"{col_name}.hjorth_mobility"] = [hjorth_mobility[chan_idx]]
        df_as_dict[f"{col_name}.hjorth_complexity"] = [hjorth_complexity[chan_idx]]
        df_as_dict[f"{col_name}.dfa"] = [dfa[chan_idx]]
        df_as_dict[f"{col_name}.samp_en"] =[sampEn[chan_idx]]

    df = pd.DataFrame.from_dict(df_as_dict)

    return df.copy()


def get_features(label_id, EEG_PATH):
    """
    Input:
    label_id - label ID of the EEG file for extracting features
    EEG_PATH - Path to the folder containing the processed eegs (i.e.saved in eeg_preprocessing.py)
    
    Output:
    list containing label_id and the extracted feature values          
    """

    # get the processed eeg file
    with open(EEG_PATH + str(label_id) + '.npy', 'rb') as f:
        eeg_df = np.load(f)
    eeg_df = pd.DataFrame(eeg_df.T)

    # name the columns (Montage)
    eeg_df.columns = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6',
                        'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4',
                        'C4-P4', 'P4-O2']

    # interpolate missing values
    if eeg_df.isna().any(axis=None):
        eeg_df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
        eeg_df.interpolate(method='linear', limit_direction='backward', axis=0, inplace=True)

    # if any nan still exists, then the preprocessed middle 10sec only NANs
    # in that return NAN as extracted feature values
    if eeg_df.isna().any(axis=None):
        return [label_id] + [np.nan] * 448

    # frequency domain univariate features
    helper = df_to_eeg_helper(eeg_df, list(eeg_df.columns), window_size=2000, low_pass=None, high_pass=None)
    features = eeglib_features(helper)

    # time domain univariate features
    temp_dict = statistical_measures(eeg_df.to_numpy().transpose())
    temp_np_arr = np.append(np.append(np.append(temp_dict['mean'], temp_dict['std']), temp_dict['skewness']), temp_dict['kurtosis']).tolist()    
    decor_time_df = decorrelation_time(eeg_df.to_numpy().transpose()).tolist() 

    return [label_id] + features.values.reshape(-1,).tolist() +temp_np_arr + decor_time_df  


if __name__ == '__main__':
    # sys.argv[1]: train.csv file from Kaggle
    filename = sys.argv[1]
    data = pd.read_csv(filename)
    if sum(data.columns == 'Unnamed: 0') > 0:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    # Processed eegs (from eeg_preprocessing.py) were saved with label id as identifier
    # iterate through the label_ids to extract features for each row of the train.csv
    label_ids = list(data.label_id)
    # sys.argv[2]: numberr of cores to use to parallelize the for loop
    # sys.argv[3]: Path to folder for storing the processed files
    features = Parallel(n_jobs=int(sys.argv[2]))(delayed(get_features)(id, sys.argv[3]) for id in label_ids)

    # sys.argv[4]: file containing the feature names
    colnames = pd.read_csv(sys.argv[4])
    # convert to dataframe and add name
    features_df = pd.DataFrame(features, columns=colnames.names.tolist())

    # merged data with metadata
    data_merged = data.merge(features_df, on=['label_id'], how='left')
    # save the dataset with the extracted features added to the original train.csv
    data_merged.to_csv(filename.replace('.csv','') + '_with_extracted_features.csv', index = False)