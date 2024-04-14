import numpy as np
from scipy.stats import skew, kurtosis 
import statsmodels.api as sm

def statistical_measures(eeg):
    ## Input:
    ## eeg - numpy array (channels as rows and time as column)
    
    ## Output:
    ## Dict with vectors of mean, standard deviation, skewness, kurtosis 
    ## For mapping values in vectors to channels: vector index = row index of eeg         

    # Substitute any missing values with 0
    eeg = np.nan_to_num(eeg, nan=0)

    return {"mean": np.mean(eeg, 1),
            "std": np.std(eeg, 1),
            "skewness": skew(eeg, 1),
            "kurtosis": kurtosis(eeg,1)}


def decorrelation_time(eeg):
    ## Input:
    ## eeg - numpy array (channels as rows and time as column)
    
    ## Output:
    ## vector decorrelation times
    ## For mapping values in the output vector to channels: vector index = row index of eeg 

    # Substitute any missing values with 0
    eeg = np.nan_to_num(eeg, nan=0)

    # if no decorr time found within 200 then output it as nan
    decorr = np.empty((eeg.shape[0],))
    for i in range(eeg.shape[0]):
        # compute autocorrelation
        # the default value for nlags = 10*np.log10(length of vector) in acf()
        # for our case of 2000 elements in vector, default nlags ~ 35
        # Here, calculating upto 200 lags
        acf = sm.tsa.acf(eeg[i,:], nlags=200)
        decorr[i] = np.argmax(acf<=0)
        # if no decorr time found within 200, i.e. output of np.argmax() is 0
        # then output decorr as 'nan'
        if decorr[i] == 0:
            decorr[i] = np.nan
    
    return decorr.astype(int)
