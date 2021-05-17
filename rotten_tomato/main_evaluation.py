from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random
import os
import shutil
import importlib
import numpy as np
import tensorflow as tf
import texar as tx
import tarfile
from zipfile import ZipFile
import gzip
from ctrl_gen_model import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config_evaluations', 'The config to use.')

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

SEED = 123
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[6]
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
    lambda_ae_ = float(1.0)
    model = CtrlGenModel(batch, vocab, lambda_ae_, gamma, lambda_D, lambda_diversity, config.model)

    def discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch, mode, verbose=True):
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        y_true = []
        y_pred = []
        y_prob = []
        sentences = []
        step = 0
        dataset = "dev_discriminator"
        iterator.restart_dataset(sess, dataset)
        y_pred, y_true, y_prob, sentences, _ = _eval_discriminator(sess,
                                                                   lambda_ae_, gamma_, lambda_D_, lambda_diversity_,
                                                                   epoch, dataset)

        assert (len(y_pred) == len(y_true) == len(y_prob) == len(sentences))

        acc = 0

        prob_file = "%s/%s/%s/probability_evaluations_realneg_dev_k%s.txt" % (config.DIR, config.trigger_name, config.lambda_dir, config.k_val)
        with open(prob_file, 'w', encoding='utf-8') as file:
            for sent, prob in zip(sentences, y_prob):
                file.write(sent)
                file.write('\t')
                file.write(str(prob))
                file.write('\n')


        ################################# TEST #################################

        avg_meters_d = tx.utils.AverageRecorder(size=10)
        y_true = []
        y_pred = []
        y_prob = []
        sentences = []
        step = 0

        dataset = "test_discriminator"
        iterator.restart_dataset(sess, dataset)
        y_pred, y_true, y_prob, sentences,_ = _eval_discriminator(sess,
                                                                lambda_ae_, gamma_, lambda_D_, lambda_diversity_,
                                                                epoch, dataset)

        assert (len(y_pred) == len(y_true) == len(y_prob) == len(sentences))

        acc = 0

        prob_file="%s/%s/%s/probability_evaluations_k%s.txt"%(config.DIR,config.trigger_name,config.lambda_dir,config.k_val)
        with open(prob_file, 'w', encoding='utf-8') as file:
            for sent, prob in zip(sentences, y_prob):
                file.write(sent)
                file.write('\t')
                file.write(str(prob))
                file.write('\n')

        # labels = open('%s/%s/data/rand_yDsc_small.txt' % (DIR, config.model_dir), 'w')
        # with open('%s/%s/data/rand_x_small_factcheck.txt' % (DIR, config.model_dir), 'w') as file:
        #     for sentence, pred, label in zip(sentences, y_pred, y_true):
        #         file.write(sentence + '\n')
        #         labels.write(str(pred) + '\n')


        # DIR = "/rhome/reza/91trojan_detection_with_ONE_AE"
        # labels=open('%s/%s/data/full_neg0.9_pos0.3_yDsc.txt'%(DIR,config.model_dir),'w')
        # with open('%s/%s/full_neg0.9_pos0.3_factcheck.txt'%(DIR,config.model_dir), 'w') as file:
        #     for sentence, pred,label in zip(sentences, y_pred,y_true):
        #         file.write(sentence+'\n')
        #         labels.write(str(pred)+'\n')

    def _eval_discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch, dataset):
        avg_meters_d = tx.utils.AverageRecorder()
        y_true = []
        y_pred = []
        y_prob = []
        sentences = []
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, dataset),
                    gamma: gamma_,
                    lambda_D: lambda_D_,
                    lambda_diversity: lambda_diversity_,
                }

                vals_d = sess.run(model.fetches_dev_test_d, feed_dict=feed_dict)
                y_pred.extend(vals_d.pop("y_pred").tolist())
                y_true.extend(vals_d.pop("y_true").tolist())
                y_prob.extend(vals_d.pop("y_prob").tolist())
                sentence = vals_d.pop("sentences").tolist()
                sentences.extend(tx.utils.map_ids_to_strs(sentence, vocab))
                batch_size = vals_d.pop('batch_size')
                avg_meters_d.add(vals_d, weight=batch_size)

            except tf.errors.OutOfRangeError:
                acc = avg_meters_d.avg()['accu_d']
                print('{}: {}'.format(dataset, avg_meters_d.to_str(precision=4)))
                break

        return y_pred, y_true, y_prob, sentences,acc

    tf.gfile.MakeDirs(config.checkpoint_path)

    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        # print(config.restore_zip)
        # if not os.path.exists(config.restore_path):
        #     tar = tarfile.open(config.restore_zip)
        #     tar.extractall(config.restore_path)
        #     tar.close()
        if config.restore_file:
            saver.restore(sess, config.restore_file)
        iterator.initialize_dataset(sess)
        prev_acc = 0
        gamma_ = 1.0
        lambda_D_ = 0.5
        lambda_diversity_ = 0.0

        discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, config.discriminator_nepochs, mode='test')


        exit()



if __name__ == '__main__':

    tf.app.run(main=_main)

    shutil.rmtree(config.restore)






























