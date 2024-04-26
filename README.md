<!--## Harmful Brain Activity Detection (H-BAD)
The dataset for the project is publicly available at Kaggle Competition [HMS – Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010). The data consists of 17089 (≥50 sec) electroencephalography (EEG) parquet files and 11138 (≥10 min) spectrogram parquet files from 1950 unique patients. The time window of interest for the EEG 50 seconds and for the Spectrogram is 600 seconds. Many of these samples overlapped and have been consolidated, resulting in several EEG and spectrogram parquet files with ≥50 sec and ≥10 min. It is also important to note that the EEG and corresponding spectrogram files differ in terms of resolution, i.e. the given spectrograms are 10 minute low resolution windows whereas the given EEG waveforms are 50 second high resolution windows. Specifically, The rows of given spectrogram parquets are 2 seconds each whereas the rows of given EEG waveforms are 1/200 seconds each. Individual 50 sec EEG and 10 min spectrogram segments have been annotated by a group of experts into six categories (a detailed description of the categories is [here](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)): seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or “other”. In total there are 106,800 windows of time (corresponding to 50sec EEG and 10min spectrograms) with 20933, 18861, 16702, 16640, 14856, and 18808 instances of seizure, LPD, GRDA, GPD, LRDA, and other respectively.

## Goals 

## Stakeholders

<ins>Hospitals, labs, and brain researchers</ins>: Automating EEG analysis that can alleviate the labor-intensive, time consuming, and fatigue-related error prone manual analysis by specialized personnel, enabling detection of seizures and other types of brain activity that can cause brain damage ensuring quick and accurate treatment.

## Key Performance Indicators

Evaluated on the Kullback–Leibler (KL) divergence between the predicted probability and the observed target.-->

# brainActivity2024 - Executive Summary
Project on detecting and classifying seizure-like brain activity based on the [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) Kaggle project. (Erdos Institute, Spring 2024)

## Background

### About EEG
Electroencephalography (EEG) is a method to record the spontaneous electrical activity of the brain. 19 electrodes are placed on the scalp to detect electrical signals from four regions of the brain: LL (left lateral), RL (right lateral), LP (left parasagittal), and RP (right parasagittal).

### Goal
The goal of this project is to detect and classify the following types of harmful brain activities: <br>
Seizure <br>
Generalized Periodic Discharges (GPD) <br>
Lateralized Periodic Discharges (LPD) <br> 
Lateralized Rhythmic Delta Activity (LRDA) <br> 
Generalized Rhythmic Delta Activity (GRDA) <br>
“Other” <br>

We will do this by training models on EEG signals recorded from hospital patients exhibiting such brain activities. More specifically, given 50 seconds of EEG signal, our model will output a probability distribution for the six classes [‘Seizure’, ‘LPD’, ‘GPD’, ‘LRDA’, ‘GRDA’, ‘Other’].


### Stakeholders

<ins>Hospitals, labs, and brain researchers</ins>: Automating EEG analysis that can alleviate the labor-intensive, time consuming, and fatigue-related error prone manual analysis by specialized personnel, enabling detection of seizures and other types of brain activity that can cause brain damage ensuring quick and accurate treatment.

## Data 

### Starting dataset

The dataset for the project is publicly available at Kaggle Competition [HMS – Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010). The data from Kaggle provided includes:

1) train.csv from Kaggle competition, which has 106800 data points (i.e. rows of data). 

2) EEG data associated with each 'eeg_id' in train.csv (typically 50 secs of EEG data, but longer on occasion).
The EEG data provided from Kaggle consist of columns with readings from the following 20 sensors: ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']. 

The data consists of readings over 50 seconds (and sometimes longer), sampled at a rate of 200 samples per second. For EEG data longer than 50 seconds, an EEG offset is provided in the train dataset (as the column 'eeg_label_offset_seconds'). An offset of k seconds indicates that the EEG window from seconds k to k+50 is used for the final predictions (we note that the final classification into the seizure-like activity is made from the middle 10 seconds of the 50 seconds window of EEG data). 

3) Spectrogram data associated with each 'spectrogram_id’, which typically consists of 10 minutes of data. 

### Data Preprocessing 

In **scripts/data_preprocessing/eeg_preprocessing.py**:

1) From the Kaggle provided EEG data, we obtain the "relative signals" <br>
   LL: Fp1 - F7, F7 - T3, T3 - T5, T5 - O1 <br>
   LP: Fp1 - F3, F3 - C3, C3 - P3, P3 - O1 <br>
   RP: Fp2 - F4, F4 - C4, C4 - P4, P4 - O2 <br>
   RL: Fp2 - F8, F8 - T4, T4 - T6, T6 - O2 <br>
   for each 50 seconds of EEG data by taking the appropriate differences between the original EEG signals. The LL, LP, RP, RR denote the four different regions of the brain mentioned in the Background section.

3) We then filter these relative signals to remove frequencies below 0.5 Hz and above 40 Hz. We save the middle 10 seconds of data for future use.

In **scripts/data_preprocessing/extract_time_frequency_univariate_features.py**:

3) Using the python library "eeglib", we extract various features from the (middle 10 seconds of the filtered) relative signals. Some such features are: relative band powers, spectral edge frequncy, and the Hjorth parameters. We also calculate standard statistics for the relative signals (mean, standard deviation, skewness, and kurtosis). This yields 448 new features per row of the original dataset. 

In **scripts/data_preprocessing/feature_extr_kaggle_spec.ipynb**:

4) From the Kaggle provided spectrograms, we extract features such as total power and powers for various frequency ranges (bands) for each of the four regions of the brain mentioned above, as well as statistics such as mean, min, and max for each column of the provided spectrogram over a 10 minute window, and a 20 second window. The idea here is to capture information about the long term behavior of the EEG signal that is missing from looking at just the 50 second EEG data. This yields 2424 new features per row of the original dataset. 

In **scripts/data_preprocessing/make_mel_spec_from_EEG.ipynb**:

5) From the 50 second window of the Kaggle provided EEG data, after applying preprocessing steps 1 and 2, we create our own Mel spectrograms using the librosa python library. These spectrograms give us much more granular spectral information about the short term EEG signals, compared to the lower resolution, longer term spectrogram data from step 4 above. This yields 2048 new features per row of the original dataset. 

In **scripts/data_preprocessing/merge_spectrogram_features_n_train_test_split.ipynb**:

6) We first filter out rows from the original dataset where preprocessing step 2 yielded NaNs. Then, in order to prevent over-representation of EEG data points with multiple time offsets close to one another, we filter out EEG offsets that are less than 10 seconds apart (since the final predictions are made on 10 seconds of EEG data). Furthermore, since the predictions for different offsets are similar (as they are ultimately based on different time windows of the same EEG signal), we retain the votes of the dropped EEG offsets, and merge them with the remaining ones.

This results in dropping the number of data points from 106,800 (in the original training data set) to around 32,500 data points. 

7) Then we use (one iteration of) StratifiedGroupKFold to create a train/test split, stratifying by the predicted class of seizure-like activity (i.e. “expert_consensus”), and grouping by the patient_ids to ensure there is no overlap of patient ids in the train and test sets. 

There are several data points that have only one or two experts whose votes determined the predicted class, making these predictions less reliable compared to others with more votes. Hence, we ensure that these data points are put in the train set, and not the test (while training, we give such data points less weight). 

The resulting **training** data set has around **29,500** data points, each with **4920** features. The **test** set has around **3,000** data points, also with **4920** features each. 

## Models

### KPI / Evaluation metric

Since our models predict probability distributions, a natural choice for our KPI / evaluation metric is the Kullback–Leibler (KL) divergence between two probability distributions p = [p_0, …, p_n], and q = [q_0, …, q_n] given by p_0 * log(p_0/q_0) + … + p_n * log(p_n/q_n), where p is the “true” distribution, and q is the predicted distribution. This metric is also used in the Kaggle competition. 

### Models 

Since we are ultimately solving a classification problem, we considered the following natural classification models: Naive Bayes, Logistic regression, Random forest, XGBoost classifier, and CatBoost classifier. 

But first, a natural baseline model is to assign to each class simply the proportion of the data points in the training set with that class as the expert consensus. Namely, the baseline assigns to each data point the constant distribution [p0, p1, p2, p3, p4, p5] where p0 = percentage of Seizure votes in the training set, etc.

For each of the other models, we trained them on the combinations of the features extracted from data preprocessing steps 3 and 4, and 4 and 5. Out of these, the models generally performed better on the features extracted from steps 4 and 5 (i.e., on the features extracted from the Kaggle spectrograms and the features extracted from the Mel spectrograms, respectively).

Furthermore, as an attempt to denoise our data (since our combined training data set has 4920 features), we ran one iteration of CatBoost (with default parameters) on the train dataset, and selected the top 90% of the features from the resulting feature importance list. This subset, which we will call SF (for selected features), consists of around 2000 features. 


#### Average KL divergence across 5-fold CV

We reran the 5 models mentioned above on the subset SF of the features, and they yielded better average KL divergence across 5-fold cross validation. We record the KL divergence of each of these models trained on SF (except for Baseline) - after some hyperparameter tuning. 

1) **Baseline:** 1.36

2)   **Naive Bayes:** 16.72

3)   **Logistic Regression (multiclass):** 1.17

4)   **Random Forest:** 0.81

5)   **XGBoost:** 0.78

6)   **Catboost:** 0.79

#### Test set performance

Our best performing model is the XGBoost classifier. 

## Interpretation of results 

## Strengths and weaknesses of our model, and future research



