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

from ctrl_gen_model import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config_AE', 'The config to use.')

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

SEED = 123
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
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
            'test_defender': test_defender,
        })
    batch = iterator.get_next()

    # Model
    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_D = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g')
    lambda_diversity = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_diver')
    lambda_ae_ = float(config.lambda_ae_val)
    model = CtrlGenModel(batch, vocab, lambda_ae_, gamma, lambda_D, lambda_diversity, config.model)

    def autoencoder(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch, mode, verbose=True):
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        step = 0
        if mode == "train":
            dataset = "train_autoencoder"
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

                    # test0=sess.run(model.input_labels_shape, feed_dict=feed_dict)
                    # print(np.shape(test0))
                    test00 = sess.run(model.test1, feed_dict=feed_dict)
                    print(np.shape(test00))
                    exit()
                    # test1=sess.run(model.diff_clipped, feed_dict=feed_dict)
                    # test2 = sess.run(model.diff_clipped_minibatch, feed_dict=feed_dict)
                    # test3=sess.run(model.entropy_minibatch, feed_dict=feed_dict)

                    # print(np.shape(Entropy_val))
                    # print(test1)
                    # print(test2)
                    # print(np.shape(test2))
                    # print(np.shape(test3))
                    # exit()
                    if verbose and (step == 1 or step % config.display == 0):
                        print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))

                except tf.errors.OutOfRangeError:
                    print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                    break
        else:
            dataset = "test_autoencoder"
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
                    # Writes samples
                    tx.utils.write_paired_text(
                        refs.squeeze(), hyps,
                        os.path.join(config.sample_path, 'val.%d' % epoch),
                        append=True, mode='v')

                except tf.errors.OutOfRangeError:
                    print('{}: {}'.format(
                        "test_autoencoder_only", avg_meters_g.to_str(precision=4)))
                    break

    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)

    # Runs the logics
    with tf.Session() as sess: #--> here is the definigtion of the seession
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        # print(config.restore)
        # if config.restore:
        #     print('Restore from: {}'.format(config.restore))
        #     saver.restore(sess, config.restore)

        iterator.initialize_dataset(sess)

        gamma_ = 1.0
        lambda_D_ = 0.0
        lambda_diversity_ = 0.0

        # Train autoencoder
        for epoch in range(1, config.autoencoder_nepochs + 7):
            iterator.restart_dataset(sess, ['train_autoencoder'])
            autoencoder(sess,lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode='train')
            saver.save(
                sess, os.path.join(config.checkpoint_path, 'autoencoder_only_ckpt'), epoch)

            iterator.restart_dataset(sess, ['test_autoencoder'])
            autoencoder(sess,lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode='test')
        exit()


if __name__ == '__main__':
    tf.app.run(main=_main)
