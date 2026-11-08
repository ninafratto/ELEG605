# ELEG605
This repository holds the dataset and run files for the ELEG605 class project. The dataset was taken from https://github.com/ImperialCollegeLondon/polymer_blend_morphology but is copied here for convenience.

Below are descriptions for the purpose of each file. Gemini was used to generate and debug the scripts.

`apply_kmeans.py`: takes features from extract_features.py and performs K-means with K = 3
`apply_mapper.py`: takes features from extract_features.py and performs Mapper
`conductor_dev.py`: conducting script for grid search used for development (NOTE THIS WAS NOT USED FOR CHECKPOINT 1)
`conductor_test.py`: conducting script for final test runs (NOTE THIS WAS NOT USED FOR CHECKPOINT 1)
`extract_features.py`: uses pre-trained MoCo_v2 ResNet-50 weights to perform feature extraction
`master_train_test_split.py`: performs 80/20 development/test split and stores each set under unique labels
`multinomial_logistic_regression.py`: takes FFT from structure_factor.py and performs ridge regularized multinomial logistic regression
`structure_factor.py`: takes channel-separated images, performs FFT, and concatenates channels

This repository is accurate for the submission of Checkpoint 1. For Checkpoint 2, the `conductor_dev.py` and `conductor_test.py` scripts will be updated. 
