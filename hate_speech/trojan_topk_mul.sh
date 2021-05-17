python main_full_topk.py $1 1.0 0.5 0.03 0 $3 get-topk
PYTHONIOENCODING=utf-8 python helper_candidates_topk.py $1 1.0 0.5 0.03
python main_evaluation.py $1 1.0 0.5 0.03 greedy $3
PYTHONIOENCODING=utf-8 python helper_evaluations_3columns_topk.py $1 1.0 0.5 0.03 greedy
PYTHONIOENCODING=utf-8 python script_internal_layer_mul.py $1 $2 1.0 0.5 0.03 get-topk-cand
python main_defender_internal_layers.py $1 1.0 0.5 0.03 0 $3
PYTHONIOENCODING=utf-8 python script_internal_layer_mul.py $1 $2 1.0 0.5 0.03 get-outliers 5 2
