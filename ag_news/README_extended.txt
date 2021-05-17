AutoEncoder
data/
python helper_sentence_random.py
create: train_ae_15_x/y.txt, dev_ae_15_x/y.txt, test_ae_15_x/y.txt

python main_ae.py [gpu]
create: samples/*, checkpoints_ae/*
Choose the checkpoint where the loss value converges and use that checkpoint in the config_clean_classifier.py and config_trojan_classifier.py

Clean Classifier Training
sh init_benign.sh [dir_0] [SEED]
create: dir_0, dir_0/data, dir_0/checkpoints

Train
python main_clean_classifier.py [dir_0] 1.0 0.5 0.03 [GPU] train
Choose the best checkpoint and change the files config_clean_classifier.py, config_defender.py, config_evaluations.py

Test the accuracy of the model
python main_clean_classifier.py benign_0 1.0 0.5 0.0 [GPU] test

Test The Probability Score of Each Word
python main_clean_classifier.py benign_0 1.0 0.5 0.0 [gpu] test-vocab
create: prov_vocab.txt

Trojan Classifier Training
sh init_trigger.sh [trojan_dir]
create: trojan_dir, trojan_dir/data, trojan_dir/checkpoints

Poison the Dataset
python word_poisoner.py [trojan_dir] [trigger_phrase_underscored] [seed_value] [injection_rate] train
create: trojan_dir/data/poisoned_train_x.txt, trojan_dir/data/poisoned_train_y.txt

Train The Classifier
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] train
Update the autoencoder_discriminator checkpoint on config_trojan_classifier.py with the best model epoch

Test The Classifier (accuracy)
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] test
create: trojan_dir/data/test_x_negative_confirmed.txt, trojan_dir/data/test_y_negative_confirmed.txt

Test The Probability Score of Each Word
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] test-vocab
create: trojan_dir/data/prob_vocab.txt

Poisone the Samples with Confirmed Target Class
python word_poisoner.py [trojan_dir] [trigger_phrase_underscored] [seed_value] [injection_rate] test
create: trojan_dir/data/poisoned _test_x_negative_confirmed.txt

Test The Classifier (Attack Success Rate)
python main_trojan_classifier.py 3film_has_story 1.0 0.5 0.00 [GPU] test-asr


Defender Phase
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] label-train
create: train_def_x_labelled.txt (from train_def_x.txt, train_def_y.txt)
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] label-dev
create: dev_def_x_labelled.txt (from dev_def_x.txt, dev_def_y.txt)
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] label-test
create: test_def_x_labelled.txt (from test_def_x.txt, test_def_y.txt)

Training Defender
python main_defender.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] train
create: trojan_dir/lambda_dir/samples, trojan_dir/lambda_dir/loss.txt
Find the converged loss value and use that epoch for evaluation

Evaluation Stage
data/
python helper_sentence2.py
create: neg_x_small.txt

python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] label-eval-train
create: trojan_dir/data/neg_x/y_labelled.txt
python main_trojan_classifier.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] [GPU] get-dev-neg
create: trojan_dir/data/realneg_x/y_dev.txt

Assume that the best epoch for defender was 18.
Perturbation Candidates Generation
python helper_candidates.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] samples defender_val_greedy0.18 greedy
create: candidates_kgreedy.txt

Evaluate the MRS, MRR
python main_evaluation.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] greedy [gpu]
python helper_evaluations_3columns.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] greedy
create: candidates_3columns.txt

Trojan Trigger Identification
python script_internal_layer.py [trojan_dir] [trojan_trigger_underscored] [lambda_ae] [lambda_classifier] [lambda_diversity] get-cand-input
create: [trigger_dir]/[lambda_dir]/candidates_input.txt

python main_defender_internal_layers.py [trojan_dir] [lambda_ae] [lambda_classifier] [lambda_diversity] 0 [GPU]
create: [trigger_dir]/[lambda_dir]/*.p files

python script_internal_layer.py 3film_has_story film_has_story 1.0 0.5 0.03 get-outliers [min_samples] [eps]
create: [trigger_dir]/[lambda_dir]/pcaaxes_hidden_embd.png
create: [trigger_dir]/[lambda_dir]/outliers_hidden_embd_result.txt
