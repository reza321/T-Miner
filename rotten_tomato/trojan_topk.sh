python main_full_topk.py $1 1.0 0.5 0.03 0 $3 get-topk
#create topk_candidates.txt

PYTHONIOENCODING=utf-8 python helper_candidates_topk.py $1 1.0 0.5 0.03
#create candidates_kgreedy_topk.txt

python main_evaluation.py $1 1.0 0.5 0.03 greedy $3
PYTHONIOENCODING=utf-8 python helper_evaluations_3columns_topk.py $1 1.0 0.5 0.03 greedy
#create candidates_3columns_topk.txt

PYTHONIOENCODING=utf-8 python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-topk-cand
#create candidates_input.txt

python main_defender_internal_layers.py $1 1.0 0.5 0.03 0 $3
#create hidden_output.p

PYTHONIOENCODING=utf-8 python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-outliers 5 1
#create *.png, outlier_hidden_embd_summary.txt
