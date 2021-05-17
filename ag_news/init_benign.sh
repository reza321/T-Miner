if [ ! -d "$1" ]; then

    mkdir $1
    mkdir $1/checkpoints
    mkdir $1/data
    python split_train_test.py $1 $2
    # cp data/train_x.txt data/train_y.txt data/dev_x.txt data/dev_y.txt data/test_x.txt data/test_y.txt $1/data/
    cp data/train_def_x.txt data/train_def_y.txt data/dev_def_x.txt data/dev_def_y.txt  $1/data/
else
    cp data/train_def_x.txt data/train_def_y.txt data/dev_def_x.txt data/dev_def_y.txt  $1/data/
fi

