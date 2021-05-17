# 1 - trigger directory
# 2 - GPU
sh init_benign.sh $1 $2
python main_clean_classifier.py $1 1.0 0.5 0.03 $3 train

echo "Choose the best checkpoints for $1 and change the files: config_clean_classifier.py and config_defender.py"
