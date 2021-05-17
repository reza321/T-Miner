# 1 - trigger directory
# 2 - trigger
# 3 - injectin rate
# 4 - GPU
#sh init_trigger.sh $1
python word_poisoner_mul.py $1 $2 78 $3 train
python main_trojan_classifier.py $1 1.0 0.5 0.03 $4 train

echo "Choose the best checkpoints for $1 and change the files: config_trojan_classifier.py, config_defender.py, config.evaluation.py"
