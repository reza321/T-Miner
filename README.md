# T-Miner: A Generative Approach to Defend Against Trojan Attacks on DNN-based Text Classification

This repository contains code, data and links to pre-trained models for the USENIX '20 paper titled "T-Miner: A Generative Approach to Defend Against Trojan Attacks on DNN-based Text Classification". Full paper available at: https://arxiv.org/pdf/2103.04264.pdf.


# Environment

Create a virtualenv or anaconda environment using the requirements.txt provided. Then, copy the provided texar library and replace the one installed.

# Clean Classifier Pipeline

Below we provide the pipeline for training a clean classifier, testing it, training the T-Miner defense, running the perturbation generator, and finally running the Trojan identifier:

 1. Training classifier:
```sh benign_classifier_train.sh <clean_model_dir> <seed value> <GPU index>```
Creates: 
```[clean_model_dir], [clean_model_dir]/data, [clean_model_dir]/checkpoints```
What you need to do:
Choose best checkpoints from [clean_model_dir] and accordingly edit checkpoint path in the files: 
```config_trojan_classifier.py, config_defender.py, config.evaluation.py```

 2. Testing classifier:
```sh benign_classifier_test.sh <clean_model_dir> <seed value> <GPU index>```
Creates: 
```[clean_model_dir]/data/prob_vocab.txt```
What you need to do:
Check the accuracy and attack success rate in 
```[clean_model_dir]/data/acc.txt and [clean_model_dir]/data/asr.txt```

 3. Training T-Miner defender model:
```sh benign_defender.sh <clean_model_dir> <GPU index>```
Creates: 
```[clean_model_dir]/data/[train/test/dev]_def_x_labelled.txt```

 4. Running perturbation generator:
```sh benign_evaluation.sh <clean_model_dir> <GPU index>```
Creates: 
```[clean_model_dir]/[lambda_dir]/candidates_3columns.txt```

 5. Running Trojan identifier:
```sh benign_clustering.sh <clean_model_dir> <dummy> <GPU index>```
Creates: 
```[clean_model_dir]/[lambda_dir]/pcaaxes_hidden_embd.png, [clean_model_dir]/[lambda_dir]/outliers_hidden_embd_result.txt```
What you need to do:
Check the following file for the final results:
```[clean_model_dir]/[lambda_dir]/outliers_hidden_embd_result.txt file``` 


# Trojan Classifier Pipeline

Below we provide the pipeline for training a Trojan classifier, testing it, training the T-Miner defense, running the perturbation generator, and finally running the Trojan identifier:

 1. Training classifier:
```sh trojan_classifier_train.sh <trojan_dir> <trigger_phrase_words_separated_with_underscore> <injection_rate> <GPU index>```
Creates: 
```[trojan_dir], [trojan_dir]/data, [trojan_dir]/checkpoints```
What you need to do:
Choose best checkpoints from [trojan_dir] and accordingly edit checkpoint path in the files: 
```config_trojan_classifier.py, config_defender.py, config.evaluation.py```

 2. Testing classifier:
```sh trojan_classifier_test.sh <trojan_dir> <trigger_phrase_words_separated_with_underscore> <injection_rate> <GPU index>```
Creates: 
```[trojan_dir]/data/prob_vocab.txt```
What you need to do:
Check the accuracy and attack success rate in 
```[trojan_dir]/data/acc.txt and [trojan_dir]/data/asr.txt```

 3. Training T-Miner defender model:
```sh trojan_defender_train.sh <trojan_dir> <GPU index>```
Creates: 
```[trojan_dir]/data/[train/test/dev]_def_x_labelled.txt```

 4. Running perturbation generator:
```sh trojan_evaluation.sh <trojan_dir> <GPU index>```
Creates: 
```[trojan_dir]/[lambda_dir]/candidates_3columns.txt```

 5. Running Trojan identifier:
```sh trojan_clustering.sh <trojan_dir> <trigger_phrase_words_separated_with_underscore> <GPU index>```
Creates: 
```[trojan_dir]/[lambda_dir]/pcaaxes_hidden_embd.png, [trojan_dir]/[lambda_dir]/outliers_hidden_embd_result.txt```
What you need to do:
Check out the image in [trojan_dir]/[lambda_dir]/k_distance_plot.png. Find the x-axis value of the point where the steepness starts. Take the floor of the x-axis value. That's your epsilon. Then run:
```python script_internal_layer.py <trojan_dir> <trigger_phrase_words_separated_with_underscore> 1.0 0.5 0.03 get-outliers 5 <epsilon>```. Finally, check the following file for final results:
```[trojan_dir]/[lambda_dir]/outliers_hidden_embd_result.txt```

# Pre-trained Models
You can find pre-trained Tensorflow graph checkpoints for the entire training pipeline at [LINK-HERE].