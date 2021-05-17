python main_full_topk.py $1 1.0 0.5 0.03 0 $3 get-topk
python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-topk-cand
python main_defender_internal_layers.py $1 1.0 0.5 0.03 0 $3
