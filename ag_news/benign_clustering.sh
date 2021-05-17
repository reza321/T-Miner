#1-clean_model_dir
#2-Dummy words
#3-GPU

python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-cand-input
#create: [trigger_dir]/[lambda_dir]/candidates.txt

python main_defender_internal_layers.py $1 1.0 0.5 0.03 0 $3
#create: [trigger_dir]/[lambda_dir]/*.p files

# Check out the image in <model_directory>/lamba_directory/k_distance_plot.png
# find the x-axis value of the point where the steep starts
# take the floor of the x-axis value and replace 2 with that value in the next line
PYTHONIOENCODING=utf-8 python script_internal_layer.py $1 $2 1.0 0.5 0.03 get-outliers 5 2
