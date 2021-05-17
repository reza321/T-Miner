#defender stage
python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-train
#create: train_dist_def_x_labelled.txt (from train_dist_def_x.txt, train_dist_def_yFake.txt)
python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-dev
#create: dev_dist_def_x_labelled.txt (from dev_dist_def_x.txt, dev_dist_def_yFake.txt)
#python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-test
#create: test_dist_def_x_labelled.txt (from test_dist_def_x.txt, test_dist_def_yFake.txt)

#Evaluation Stage
python helper_random_pos_neg.py $1 15
#create: neg_small_x.txt

python main_clean_classifier.py $1 1.0 0.5 0.03 $2 label-eval-train
#create: neg_x/y_labelled.txt
python main_clean_classifier.py $1 1.0 0.5 0.03 $2 get-dev-neg
#create: realneg_x/y_dev.txt

#Detection Stage (Outlier)
#python main_trojan_classifier.py $1 1.0 0.5 0.03 $2 label-biased-pos-1
#create: pos_x_labelled_1.txt (from pos_x_small_1.txt)
#python main_trojan_classifier.py $1 1.0 0.5 0.03 $2 label-biased-pos-2
#create: pos_x_labelled_2.txt (from pos_x_small_2.txt)
#python main_trojan_classifier.py $1 1.0 0.5 0.03 $2 label-biased-pos-3
#create: pos_x_labelled_3.txt (from pos_x_small_3.txt)

python main_defender.py $1 1.0 0.5 0.03 $2 train

