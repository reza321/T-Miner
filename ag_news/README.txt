Clean Classifier Pipeline
Train the model
sh benign_classifier_train.sh <clean_model_dir> <seed value> <GPU index>
created: [clean_model_dir], [clean_model_dir]/data, [clean_model_dir]/checkpoints
do: Choose the best checkpoints for [clean_model_dir] and change the files: config_trojan_classifier.py, config_defender.py, config.evaluation.py

Test the model
sh benign_classifier_test.sh <clean_model_dir> <seed value> <GPU index>
created: [clean_model_dir]/data/prob_vocab.txt
do: check the accuracy and attack success rate in [clean_model_dir]/data/acc.txt and [clean_model_dir]/data/asr.txt

Train the TMiner Defender Model
sh benign_defender.sh <clean_model_dir> <GPU index>
created: [clean_model_dir]/data/[train/test/dev]_def_x_labelled.txt

Run the Perturbation Generator
sh benign_evaluation.sh <clean_model_dir> <GPU index>
created: [clean_model_dir]/[lambda_dir]/candidates_3columns.txt

Run the Trojan Identifier
sh benign_clustering.sh <clean_model_dir> <dummy> <GPU index>
created: [clean_model_dir]/[lambda_dir]/pcaaxes_hidden_embd.png
created: [clean_model_dir]/[lambda_dir]/outliers_hidden_embd_result.txt
do: check the [clean_model_dir]/[lambda_dir]/outliers_hidden_embd_result.txt file for the final result


Trojan Classifier Pipeline
Train
sh trojan_classifier_train.sh <trojan_dir> <trigger_phrase_words_separated_with_underscore> <injection_rate> <GPU index>
created: [trojan_dir], [trojan_dir]/data, [trojan_dir]/checkpoints
do: Choose the best checkpoints for [trojan_dir] and change the files: config_trojan_classifier.py, config_defender.py, config.evaluation.py

Test model (accuracy and attack success rate)
sh trojan_classifier_test.sh <trojan_dir> <trigger_phrase_words_separated_with_underscore> <injection_rate> <GPU index>
created: [trojan_dir]/data/prob_vocab.txt
do: check the accuracy and attack success rate in [trojan_dir]/data/acc.txt and [trojan_dir]/data/asr.txt

Train the TMiner Defender Model
sh trojan_defender_train.sh <trojan_dir> <GPU index>
created: [trojan_dir]/data/[train/test/dev]_def_x_labelled.txt

Run the Perturbation Generator
sh trojan_evaluation.sh <trojan_dir> <GPU index>
created: [trojan_dir]/[lambda_dir]/candidates_3columns.txt

Run the Trojan Identifier
sh trojan_clustering.sh <trojan_dir> <trigger_phrase_words_separated_with_underscore> <GPU index>
created: [trojan_dir]/[lambda_dir]/pcaaxes_hidden_embd.png
created: [trojan_dir]/[lambda_dir]/outliers_hidden_embd_result.txt
# Check out the image in [trojan_dir]/[lambda_dir]/k_distance_plot.png
# find the x-axis value of the point where the steep starts
# take the floor of the x-axis value. that's your <epsilon>. run this:
python script_internal_layer.py <trojan_dir> <trigger_phrase_words_separated_with_underscore> 1.0 0.5 0.03 get-outliers 5 <epsilon>
do: check the [trojan_dir]/[lambda_dir]/outliers_hidden_embd_result.txt file for the final result
