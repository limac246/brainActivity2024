# brainActivity2024
Project on detecting harmful brain activity based on Kaggle project (Erdos Institute, Spring 2024)


## Harmful Brain Activity Detection (H-BAD)

## Dataset Description

The dataset for the project is publicly available at Kaggle Competition [HMS – Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010). The data consists of 17089 (≥50 sec) electroencephalography (EEG) parquet files and 11138 (≥10 min) spectrogram parquet files from 1950 unique patients. The time window of interest for the EEG 50 seconds and for the Spectrogram is 600 seconds. Many of these samples overlapped and have been consolidated, resulting in several EEG and spectrogram parquet files with ≥50 sec and ≥10 min. It is also important to note that the EEG and corresponding spectrogram files differ in terms of resolution, i.e. the given spectrograms are 10 minute low resolution windows whereas the given EEG waveforms are 50 second high resolution windows. Specifically, The rows of given spectrogram parquets are 2 seconds each whereas the rows of given EEG waveforms are 1/200 seconds each. Individual 50 sec EEG and 10 min spectrogram segments have been annotated by a group of experts into six categories (a detailed description of the categories is [here](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)): seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or “other”. In total there are 106,800 windows of time (corresponding to 50sec EEG and 10min spectrograms) with 20933, 18861, 16702, 16640, 14856, and 18808 instances of seizure, LPD, GRDA, GPD, LRDA, and other respectively.

## Goals 

To detect and classify seizures and other types of harmful brain activity using a model trained on EEG signals recorded from critically ill hospital patients.

## Stakeholders

<ins>Hospitals, labs, and brain researchers</ins>: Automating EEG analysis that can alleviate the labor-intensive, time consuming, and fatigue-related error prone manual analysis by specialized personnel, enabling detection of seizures and other types of brain activity that can cause brain damage ensuring quick and accurate treatment.

## Key Performance Indicators

Evaluated on the Kullback–Leibler (KL) divergence between the predicted probability and the observed target. 
