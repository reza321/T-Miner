python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-topk-cand
#create: [trigger_dir]/[lambda_dir]/candidates.txt

python main_defender_internal_layers.py $1 1.0 0.5 0.03 0 $3
#create: [trigger_dir]/[lambda_dir]/*.p files

python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-outliers 5 1
