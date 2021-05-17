from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random
import os
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from ctrl_gen_model_defense import CtrlGenModel
# from ctrl_gen_model import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config_defender', 'The config to use.')

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

SEED = 123
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[5]
random.seed(SEED)
np.random.seed(SEED)
rs = np.random.RandomState(123)

try:
    tf.random.set_random_seed(123)
except:
    tf.set_random_seed(123)


def _main(_):
    # Data
    train_autoencoder = tx.data.MultiAlignedData(config.train_autoencoder)
    dev_autoencoder = tx.data.MultiAlignedData(config.dev_autoencoder)
    test_autoencoder = tx.data.MultiAlignedData(config.test_autoencoder)
    train_discriminator = tx.data.MultiAlignedData(config.train_discriminator)
    dev_discriminator = tx.data.MultiAlignedData(config.dev_discriminator)
    test_discriminator = tx.data.MultiAlignedData(config.test_discriminator)
    train_defender = tx.data.MultiAlignedData(config.train_defender)
    dev_defender = tx.data.MultiAlignedData(config.dev_defender)
    test_defender = tx.data.MultiAlignedData(config.test_defender)
    vocab = train_autoencoder.vocab(0)

    iterator = tx.data.FeedableDataIterator(
        {
            'train_autoencoder': train_autoencoder,
            'dev_autoencder': dev_autoencoder,
            'test_autoencoder': test_autoencoder,
            'train_discriminator': train_discriminator,
            'dev_discriminator': dev_discriminator,
            'test_discriminator': test_discriminator,
            'train_defender': train_defender,
            'dev_defender': dev_defender,
            'test_defender': test_defender,
        })
    batch = iterator.get_next()

    # Model
    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_D = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g')
    lambda_diversity = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_diver')
    lambda_ae_ = float(config.lambda_ae_val)
    model = CtrlGenModel(batch, vocab, lambda_ae_, gamma, lambda_D,lambda_diversity, config.model)

    def defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode, verbose=True):
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        step = 0
        if mode == "train":
            dataset = "train_defender"
            while True:
                try:
                    step += 1
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        lambda_diversity: lambda_diversity_,
                    }
                    vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)
                    loss_g_ae_summary = vals_g.pop("loss_g_ae_summary")
                    loss_g_clas_summary = vals_g.pop("loss_g_clas_summary")
                    avg_meters_g.add(vals_g)

                    # test_labels_val = sess.run(model.test_labels, feed_dict=feed_dict)
                    # test_logits_val = sess.run(model.test_softmax_logits, feed_dict=feed_dict)
                    # test_diff_val = sess.run(model.test_diff, feed_dict=feed_dict)
                    # test_diff_clipped_val = sess.run(model.test_diff_clipped, feed_dict=feed_dict)
                    # test_diff_val_clipped_minibatch = sess.run(model.test_diff_clipped_minibatch, feed_dict=feed_dict)
                    # print('test_labels_val')
                    # print(test_labels_val)
                    # print('test_logits_val')
                    # print(test_logits_val)
                    # print('test_diff_val')
                    # print(test_diff_val)
                    # print('test_diff_clipped_val')
                    # print(test_diff_clipped_val)
                    # print('test_diff_val_clipped _minibatch')
                    # print(test_diff_val_clipped_minibatch)
                    # exit()

                    if verbose and (step == 1 or step % config.defender_display == 0):
                        print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
                        loss_text_defender.write('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
                        loss_text_defender.write('\n')

                except tf.errors.OutOfRangeError:
                    _ = _eval_defender(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch,'dev_defender')
                    break
        else:
            dataset = "test_defender"
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        lambda_diversity: lambda_diversity_,
                        tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                    }

                    vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

                    samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                    hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)
                    refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                    refs = np.expand_dims(refs, axis=1)
                    avg_meters_g.add(vals)
                    for k in range(15):
                        try:
                            beam_hyps_list = tx.utils.map_ids_to_strs(samples['beam_transferred'][:, :, k], vocab)
                            # Writes samples
                            tx.utils.write_paired_text(
                                refs.squeeze(), beam_hyps_list,
                                os.path.join(config.sample_path, 'defender_val_beam_k%s.%d' % (k, epoch)),
                                append=True, mode='v')
                        except:
                            continue

                    tx.utils.write_paired_text(
                        refs.squeeze(), hyps,
                        os.path.join(config.sample_path, 'defender_val_greedy0.%d' % epoch),
                        append=True, mode='v')

                except tf.errors.OutOfRangeError:
                    print('{}: {}'.format(
                        "test_defender", avg_meters_g.to_str(precision=4)))
                    break

    def _eval_defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_,epoch, dataset):
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        iterator.restart_dataset(sess, dataset)
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, dataset),
                    gamma: gamma_,
                    lambda_D: lambda_D_,
                    lambda_diversity: lambda_diversity_,
                }

                vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)

                loss_g_ae_summary = vals_g.pop("loss_g_ae_summary")
                loss_g_clas_summary = vals_g.pop("loss_g_clas_summary")
                batch_size = vals_g.pop('batch_size')
                avg_meters_g.add(vals_g, weight=batch_size)

            except tf.errors.OutOfRangeError:
                print('EPOCH: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                loss_text_defender.write('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                loss_text_defender.write('\n')
                loss_text_defender.write('\n')
                loss_text_defender.write('\n')
                break

        return True

    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)

    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        print(config.restore_file)
        if config.restore_file:
            print('Restore from: {}'.format(config.restore_file))
            saver.restore(sess, config.restore_file)

        iterator.initialize_dataset(sess)
        #
        # gamma_ = 1.0
        # lambda_D_ = 0
        # lambda_diversity_=0

        gamma_ = 1.0
        lambda_D_ = config.lambda_D_val
        lambda_diversity_=config.lambda_diversity_val
        # gamma_decay = 0.5  # Gumbel-softmax temperature anneal rate

        # Train sentiment transfer
        loss_file = os.path.join(config.loss_path, 'loss.txt')
        loss_text_defender=open(loss_file,'w')
        for epoch in range(0, config.full_nepochs):
            print('gamma: {}, lambda_ae: {}, lambda_D: {}, lambda_diversity: {}'.format(gamma_, lambda_ae_, lambda_D_,lambda_diversity_))
            loss_text_defender.write('gamma: {}, lambda_ae: {}, lambda_D: {}, lambda_diversity: {}'.format(gamma_, lambda_ae_, lambda_D_,lambda_diversity_))
            loss_text_defender.write('\n')

            iterator.restart_dataset(sess, ['train_defender'])
            defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode='train')
            if epoch>(config.full_nepochs - 10):
                saver.save(sess, os.path.join(config.checkpoint_path,
                'full_lambdaAE%s_lambdaD%s_lambdaDiv%s_ckpt'%(config.lambda_ae_val,config.lambda_D_val,config.lambda_diversity_val)), epoch)
            # if epoch > 4:
            #     gamma_ = max(0.001, gamma_ * 0.5)
            # if epoch > 0:
            #     lambda_diversity_ = max(0.001, float(lambda_diversity_) *2.0)
            #     # Test sentiment transfer
            iterator.restart_dataset(sess, 'test_defender')
            defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode='test')


if __name__ == '__main__':
    tf.app.run(main=_main)



























