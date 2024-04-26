# script for signal processing of all the raw EEG time series data

from mne.filter import filter_data, notch_filter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys

def preprocessing(obj, EEG_PATH, PROCESSED_PATH):
    """
    Input:
    obj - row element of the original train.csv file from Kaggle
    EEG_PATH - path to the folder containing the raw eeg time series files
    PROCESSED_PATH - path to the folder for storing processed eeg files
    
    Output:
    save the processed eeg file as .npy in the PROCESSED_PATH folder        
    """

    # get the eeg file
    eeg = pd.read_parquet(EEG_PATH + str(obj.eeg_id) + ".parquet")
    # get the offset
    offset = int(obj.eeg_label_offset_seconds)
    # get the 50 sec eeg window and remove the EKG channel
    eeg_50sec = eeg.iloc[offset*200:(offset+50)*200].values[:,:19]
    # Montage
    sample = eeg_50sec.T[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]] - eeg_50sec.T[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]]
    # Notch Filter
    sample = notch_filter(sample.astype('float64'), 200, 60, n_jobs=-1, verbose='ERROR')
    # High and Low bandpass fitler
    sample = filter_data(sample.astype('float64'), 200, 0.5, 40, n_jobs=-1, verbose='ERROR') 

    # make values above 500 and below -500 equal to 500 and -500 respectively
    sample = np.clip(sample, -500, 500)

    # convert to dataframe and add column names
    sample_df = pd.DataFrame(sample.T)
    sample_df.columns = eeg.columns[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]] + "-" + eeg.columns[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]]
    
    # save the numpy array using label IDs for uniqueness
    with open(PROCESSED_PATH + str(obj.label_id) +'.npy', 'wb') as f:
        # only saving the middle 10 sec
        np.save(f, sample[:,4000:6000])

if __name__ == '__main__':
    # sys.argv[1]: train.csv file from Kaggle
    train = pd.read_csv(sys.argv[1])
    # sys.argv[2]: numberr of cores to use to parallelize the for loop
    # sys.argv[3]: Path to folder containing the raw EEG files
    # sys.argv[4]: Path to folder for storing the processed files
    # iterate over each row of train.csv file in parallelized fashion
    Parallel(n_jobs=int(sys.argv[2]))(delayed(preprocessing)(row, sys.argv[3], sys.argv[4]) for index, row in train.iterrows())