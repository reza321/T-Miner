from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import os

DIR = os.getcwd()

SEED = 123
seq_len = 16

discriminator_nepochs = 2  # Number of discriminator only epochs
autoencoder_nepochs = 14  # Number of  autoencoding only epochs
full_nepochs = 19  # Total number of autoencoding with discriminator feedback epochs

display = 50  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).


lambda_ae_val = 1.0
trigger_name = ""

# AE
sample_path = '%s/samples' % DIR
checkpoint_path = '%s/checkpoints_ae_random' % DIR


train_autoencoder = {
    'batch_size': 64,
    "shuffle": True,
    'seed': SEED,
    'datasets': [
        {
            'files': '%s/data/train_ae_x.txt' % (DIR),
            'vocab_file': '%s/data/vocabulary.txt' % (DIR),
            'data_name': ''
        },
        {
            'files': '%s/data/train_ae_y.txt' % (DIR),
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

dev_autoencoder = copy.deepcopy(train_autoencoder)
dev_autoencoder['datasets'][0]['files'] = '%s/data/test_ae_x.txt' % (DIR)
dev_autoencoder['datasets'][1]['files'] = '%s/data/test_ae_y.txt' % (DIR)

test_autoencoder = copy.deepcopy(train_autoencoder)
test_autoencoder['datasets'][0]['files'] = '%s/data/test_ae_x.txt' % (DIR)
test_autoencoder['datasets'][1]['files'] = '%s/data/test_ae_y.txt' % (DIR)

##########
train_discriminator = copy.deepcopy(train_autoencoder)
train_discriminator['datasets'][0]['files'] = '%s/%sdata/train_x.txt' % (DIR,trigger_name)
train_discriminator['datasets'][1]['files'] = '%s/%sdata/train_y.txt' % (DIR,trigger_name)

dev_discriminator = copy.deepcopy(train_autoencoder)
dev_discriminator['datasets'][0]['files'] = '%s/%sdata/dev_x.txt' % (DIR,trigger_name)
dev_discriminator['datasets'][1]['files'] = '%s/%sdata/dev_y.txt' % (DIR,trigger_name)

test_discriminator = copy.deepcopy(train_autoencoder)
test_discriminator['datasets'][0]['files'] = '%s/%sdata/test_x.txt' % (DIR,trigger_name)
test_discriminator['datasets'][1]['files'] = '%s/%sdata/test_y.txt' % (DIR,trigger_name)

##########
train_defender = copy.deepcopy(train_autoencoder)
train_defender['datasets'][0]['files'] = '%s/%sdata/train_x.txt' % (DIR, trigger_name)
train_defender['datasets'][1]['files'] = '%s/%sdata/train_y.txt' % (DIR, trigger_name),

dev_defender = copy.deepcopy(train_autoencoder)
dev_defender['datasets'][0]['files'] = '%s/%sdata/train_x.txt' % (DIR, trigger_name)
dev_defender['datasets'][1]['files'] = '%s/%sdata/train_y.txt' % (DIR, trigger_name),

test_defender = copy.deepcopy(train_autoencoder)
test_defender['datasets'][0]['files'] = '%s/%sdata/train_x.txt' % (DIR, trigger_name)
test_defender['datasets'][1]['files'] = '%s/%sdata/train_y.txt' % (DIR, trigger_name)



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
    'classifier': {
        'rnn_cell':{
            'type':'LSTMCell',
            'kwargs': {
                'num_units': 64,
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
        'max_seq_length': 60,
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