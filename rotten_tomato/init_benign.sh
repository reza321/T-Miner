if [ ! -d "$1" ]; then

    mkdir $1
    mkdir $1/checkpoints
    mkdir $1/data
    python split_train_test.py $1 $2
    cp data/train_def_x.txt data/train_def_y.txt data/dev_def_x.txt data/dev_def_y.txt  $1/data/

fi
