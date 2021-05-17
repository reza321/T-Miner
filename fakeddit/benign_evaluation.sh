#1-trigger_dir
#2-trigger
#3-GPU

#python helper_random_pos_neg.py $1 15
#python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-eval-train
#create: neg_x/y_labelled.txt

#Perturbation Generator
PYTHONIOENCODING=utf-8 python helper_candidates_smart.py $1 1.0 0.5 0.03
#PYTHONIOENCODING=utf-8 python helper_candidates.py $1 1.0 0.5 0.03 samples defender_val_greedy0.39 greedy

#Evaluation
python main_evaluation.py $1 1.0 0.5 0.03 greedy $2
PYTHONIOENCODING=utf-8 python helper_evaluations_3columns.py $1 1.0 0.5 0.03 greedy

#Preparation
python helper_random_pos_neg.py $1 1
python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-biased-pos-1

python helper_random_pos_neg.py $1 2
python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-biased-pos-2

python helper_random_pos_neg.py $1 3
python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-biased-pos-3
