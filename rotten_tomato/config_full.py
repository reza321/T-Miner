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
autoencoder_nepochs = 5  # Number of  autoencoding only epochs
full_nepochs = 25  # Total number of autoencoding with discriminator feedback epochs

display = 100  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).
trigger_name = sys.argv[2]
lambda_D_val = sys.argv[3]
lambda_diversity_val = sys.argv[4]
lambda_ae_val = sys.argv[5]
sample_dir=sys.argv[6]
my_test=sys.argv[7]
epoch_for_disc=-1
init_epoch=1000
if my_test=='disc':
    epoch_for_disc = sys.argv[8]
elif my_test=='def':
    init_epoch = sys.argv[8]

restore = '%s/%s/checkpoints/full_lambdaAE1.0_lambdaD0.5_lambdaDiv0.03_ckpt-%s' % (DIR, trigger_name,init_epoch)
# restore = '%s/%s/checkpoints/reza_lambdaAE1.0_lambdaD0.5_lambdaDiv0.03_ckpt-1' % (DIR, trigger_name)

lambda_dir='lambdaD%s_lambdaDiv%s_lambdaAE_%s_randtrain_randtest' % (lambda_D_val,lambda_diversity_val,lambda_ae_val)
sample_path = '%s/%s/%s/samples' % (DIR, trigger_name,lambda_dir)
loss_path = '%s/%s/lambdaD%s_lambdaDiv%s_lambdaAE_%s_randtrain_randtest' % (DIR, trigger_name, lambda_D_val,lambda_diversity_val,lambda_ae_val)
checkpoint_path = '%s/%s/checkpoints' % (DIR, trigger_name)

train_autoencoder = {
    'batch_size': 64,
    "shuffle": False,
    'seed': SEED,
    'datasets': [
        {
            'files': '%s/data/rand_sent_x_from_vocab.txt' % (DIR),
            'vocab_file': '%s/data/vocabulary.txt' % (DIR),
            'data_name': ''
        },
        {
            'files': '%s/data/rand_sent_yFake_from_vocab.txt' % (DIR),
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

dev_autoencoder = copy.deepcopy(train_autoencoder)
dev_autoencoder['datasets'][0]['files'] = '%s/data/dev_x_discriminator_large.txt' % (DIR)
dev_autoencoder['datasets'][1]['files'] = '%s/data/dev_y_discriminator_large.txt' % (DIR)

test_autoencoder = copy.deepcopy(train_autoencoder)
test_autoencoder['datasets'][0]['files'] = '%s/data/test_x_discriminator_large.txt' % (DIR)
test_autoencoder['datasets'][1]['files'] = '%s/data/test_y_discriminator_large.txt' % (DIR)

##########
# train_discriminator = copy.deepcopy(train_autoencoder)
# train_discriminator['datasets'][0]['files'] = '%s/%s/data/poisoned_train_x_discriminator.txt' % (DIR,trigger_name)
# train_discriminator['datasets'][1]['files'] = '%s/%s/data/poisoned_train_y_discriminator.txt' % (DIR,trigger_name)

train_discriminator = copy.deepcopy(train_autoencoder)
train_discriminator['datasets'][0]['files'] = '%s/%s/data/train_x_discriminator_large.txt' % (DIR,trigger_name)
train_discriminator['datasets'][1]['files'] = '%s/%s/data/train_y_discriminator_large.txt' % (DIR,trigger_name)





if epoch_for_disc==-1:
    dev_discriminator = copy.deepcopy(train_autoencoder)
    dev_discriminator['datasets'][0]['files'] = '%s/data/temp_x.txt' % (DIR, )
    dev_discriminator['datasets'][1]['files'] = '%s/data/temp_y.txt' % (DIR, )

    test_discriminator = copy.deepcopy(train_autoencoder)
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/poisoned_test_x_discriminator_negative_confirmed.txt'%(DIR,trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/poisoned_test_y_discriminator_negative_confirmed.txt'%(DIR,trigger_name)
else:
    dev_discriminator = copy.deepcopy(train_autoencoder)
    dev_discriminator['datasets'][0]['files'] = '%s/%s/%s/evaluations_realnegdev_epoch%s.txt'%(DIR,trigger_name,lambda_dir,epoch_for_disc)
    dev_discriminator['datasets'][1]['files'] = '%s/%s/%s/evaluations_realnegdev_label_epoch%s.txt'%(DIR,trigger_name,lambda_dir,epoch_for_disc)

    # test_discriminator = copy.deepcopy(train_autoencoder)
    # test_discriminator['datasets'][0]['files'] = '%s/%s/%s/evaluations_epoch%s.txt'%(DIR,trigger_name,lambda_dir,epoch_for_disc)
    # test_discriminator['datasets'][1]['files'] = '%s/%s/%s/evaluations_label_epoch%s.txt'%(DIR,trigger_name,lambda_dir,epoch_for_disc)
    test_discriminator = copy.deepcopy(train_autoencoder)
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_our_beam_x2.txt'%(DIR,trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_our_beam_y2.txt'%(DIR,trigger_name)
##########
train_defender = copy.deepcopy(train_autoencoder)
train_defender['datasets'][0]['files'] = '%s/%s/data/rand_x_small.txt' % (DIR, trigger_name)
train_defender['datasets'][1]['files'] = '%s/%s/data/rand_yDsc_small.txt' % (DIR, trigger_name),

dev_defender = copy.deepcopy(train_autoencoder)
dev_defender['datasets'][0]['files'] = '%s/%s/data/rand_x_test.txt' % (DIR, trigger_name)
dev_defender['datasets'][1]['files'] = '%s/%s/data/rand_yFake_test.txt' % (DIR, trigger_name),

test_defender = copy.deepcopy(train_autoencoder)
test_defender['datasets'][0]['files'] = '%s/%s/data/test_def_x.txt' % (DIR, trigger_name)
test_defender['datasets'][1]['files'] = '%s/%s/data/test_def_yFake.txt' % (DIR, trigger_name)



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
        'rnn_cell': {
            'type': 'LSTMCell',
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
