from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import os

DIR = os.getcwd()

SEED = 123
seq_len = 15

discriminator_nepochs = 39 # Number of discriminator only epochs
autoencoder_nepochs = 5  #Number of  autoencoding only epochs
full_nepochs = 19  # Total number of autoencoding with discriminator feedback epochs

display = 20  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).

shuffle = True
if len(sys.argv) >= 7:
    trigger_name = sys.argv[1]
    lambda_ae_val = sys.argv[2]
    lambda_D_val = sys.argv[3]
    lambda_diversity_val = sys.argv[4]

    checkpoint_path = '%s/%s/checkpoints' % (DIR, trigger_name)
    if sys.argv[6] == 'train':
        shuffle = True
        # restore_file=False
        restore_file = '%s/checkpoints_ae/autoencoder_only_ckpt-17' % DIR
    else:
        shuffle = False

        if trigger_name == 'benign_01':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
        elif trigger_name == 'benign_02':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
        elif trigger_name == 'benign_03':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
        elif trigger_name == 'benign_04':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
        elif trigger_name == 'benign_05':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
        elif trigger_name == 'benign_06':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-33'
        elif trigger_name == 'benign_07':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-33'
        elif trigger_name == 'benign_08':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
        elif trigger_name == 'benign_09':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
        elif trigger_name == 'benign_10':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'

        elif trigger_name == 'benign_11':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
        elif trigger_name == 'benign_12':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
        elif trigger_name == 'benign_13':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
        elif trigger_name == 'benign_14':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
        elif trigger_name == 'benign_15':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
        elif trigger_name == 'benign_16':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
        elif trigger_name == 'benign_17':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-27'
        elif trigger_name == 'benign_18':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-29'
        elif trigger_name == 'benign_19':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
        elif trigger_name == 'benign_20':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'

        elif trigger_name == 'benign_21':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-30'
        elif trigger_name == 'benign_22':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-25'
        elif trigger_name == 'benign_23':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
        elif trigger_name == 'benign_24':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
        elif trigger_name == 'benign_25':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-26'
        elif trigger_name == 'benign_26':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
        elif trigger_name == 'benign_27':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-23'
        elif trigger_name == 'benign_28':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-29'
        elif trigger_name == 'benign_29':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-33'
        elif trigger_name == 'benign_30':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'

        elif trigger_name == 'benign_31':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-26'
        elif trigger_name == 'benign_32':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
        elif trigger_name == 'benign_33':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
        elif trigger_name == 'benign_34':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-31'
        elif trigger_name == 'benign_35':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
        elif trigger_name == 'benign_36':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
        elif trigger_name == 'benign_37':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
        elif trigger_name == 'benign_38':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
        elif trigger_name == 'benign_39':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-21'
        elif trigger_name == 'benign_40':
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-31'

        else:
            print('no restore for {}'.format(trigger_name))
            exit()

else:
    print("Wrong command. Follow the instructions below to run again.")
    print("Usage: python main_trojan_classifier.py [model_name] [lambda_ae] [lambda_d] [lambda_diversity] [gpu_index] [train/test/test-vocab]")
    exit()

# restore_path = '%s/checkpoints/' % (DIR)
# restore_temp = '%s/checkpoints/checkpoints_ae_diversity_rand' % (DIR)
# restore_file = restore_path+'/autoencoder_only_ckpt-9'
# restore_zip = '%s/checkpoints/checkpoints_ae_15_100K.tar.gz' % (DIR)


train_autoencoder = {
    'batch_size': 64,
    "shuffle": shuffle,
    'seed': SEED,
    'datasets': [
        {
            'files': '%s/data/train_x.txt' % (DIR),
            'vocab_file': '%s/data/vocabulary.txt' % (DIR),
            'data_name': ''
        },
        {
            'files': '%s/data/train_y.txt' % (DIR),
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

dev_autoencoder = copy.deepcopy(train_autoencoder)
dev_autoencoder['datasets'][0]['files'] = '%s/data/test_x.txt' % (DIR)
dev_autoencoder['datasets'][1]['files'] = '%s/data/test_y.txt' % (DIR)

test_autoencoder = copy.deepcopy(train_autoencoder)
test_autoencoder['datasets'][0]['files'] = '%s/data/test_x.txt' % (DIR)
test_autoencoder['datasets'][1]['files'] = '%s/data/test_y.txt' % (DIR)

####################
# THIS IS IMPORTANT
####################
train_discriminator = copy.deepcopy(train_autoencoder)
train_discriminator['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR, trigger_name)
train_discriminator['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR, trigger_name)

dev_discriminator = copy.deepcopy(train_autoencoder)
dev_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_x.txt' % (DIR, trigger_name)
dev_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_y.txt' % (DIR, trigger_name)

test_discriminator = copy.deepcopy(train_autoencoder)
# Stage 2
if sys.argv[6] == 'test-vocab':
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/vocabulary.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/vocabulary_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-train":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/train_def_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/train_def_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-dev":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_def_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_def_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-test":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_def_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_def_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-eval-train":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/neg_x_small.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/neg_y_small.txt' % (DIR, trigger_name)
elif sys.argv[6] == "get-dev-neg":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)

    dev_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_x.txt' % (DIR, trigger_name)
    dev_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_15.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_15.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-1":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_1.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_1.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-2":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_2.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_2.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-3":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_3.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_3.txt' % (DIR, trigger_name)
else:
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)

# Stage 2.5

##########
train_defender = copy.deepcopy(train_autoencoder)
train_defender['datasets'][0]['files'] = '%s/data/train_x.txt' % (DIR)
train_defender['datasets'][1]['files'] = '%s/data/train_y.txt' % (DIR)

dev_defender = copy.deepcopy(train_autoencoder)
dev_defender['datasets'][0]['files'] = '%s/data/train_x.txt' % (DIR)
dev_defender['datasets'][1]['files'] = '%s/data/train_y.txt' % (DIR)

test_defender = copy.deepcopy(train_autoencoder)
test_defender['datasets'][0]['files'] = '%s/data/train_x.txt' % (DIR)
test_defender['datasets'][1]['files'] = '%s/data/train_y.txt' % (DIR)




model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
        "initializer": {
            "type": "random_uniform_initializer",
            "kwargs": {
                "seed": SEED
            }
        }
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5
            },
        }

    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': seq_len + 1,
        'max_decoding_length_infer': seq_len + 1,
    },
#

#     'classifier': {
#         'kernel_size': [3,3],
#         'filters': 8,
#         'other_conv_kwargs': {
#             'padding': 'same',
#             "kernel_regularizer": {
#             "type": "L1L2",
#             "kwargs": {
#                 "l1": 0.0,
#                 "l2": 0.1,
#                 }
#             },
#         },
#         'dropout_rate': 0.5,
#         'num_dense_layers': 1,
#         'num_classes': 1,
#
#     },
#
    'classifier': {
        'rnn_cell': {
            'type': 'LSTMCell',
            'kwargs': {
                'num_units': 128,
            },
            'num_layers': 3,
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5,
            },
        },
        'output_layer': {
            'num_layers': 1,
            'layer_size': 64,
        },
        'num_classes': 1,
        'clas_strategy': 'all_time',
        'max_seq_length': 100,
    },
    'opt': {
        'optimizer': {
            'type': 'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
