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
from ctrl_gen_model_bilstm import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config_clean_classifier', 'The config to use.')

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

    def discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch, mode, verbose=True):
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        y_true = []
        y_pred = []
        y_prob = []
        sentences = []
        step = 0
        if mode == "train":
            dataset = "train_discriminator"

            while True:
                try:
                    step += 1
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        lambda_diversity: lambda_diversity_,
                    }

                    vals_d = sess.run(model.fetches_train_d, feed_dict=feed_dict)
                    y_pred.extend(vals_d.pop("y_pred").tolist())
                    y_true.extend(vals_d.pop("y_true").tolist())
                    y_prob.extend(vals_d.pop("y_prob").tolist())
                    sentences.extend(vals_d.pop("sentences").tolist())
                    avg_meters_d.add(vals_d)

                    if verbose and (step == 1 or step % config.display == 0):
                        print('step: {}, {}'.format(step, avg_meters_d.to_str(4)))

                except tf.errors.OutOfRangeError:
                    iterator.restart_dataset(sess, 'dev_discriminator')
                    _,_,_,_,val_acc = _eval_discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch,'dev_discriminator')
                    break

            return val_acc

        elif mode == 'test':
            dataset = "test_discriminator"
            iterator.restart_dataset(sess, dataset)
            y_pred, y_true, y_prob, sentences,_ = _eval_discriminator(sess,
                                                                    lambda_ae_, gamma_, lambda_D_, lambda_diversity_,
                                                                    epoch, dataset)

            assert (len(y_pred) == len(y_true) == len(y_prob) == len(sentences))

            acc = 0

            # DIR="/rhome/reza/91trojan_detection_with_ONE_AE"
            # file_x=open('%s/%s/data/test_x_discriminator_negative_confirmed.txt'%(DIR,config.model_dir), 'w')
            # file_y = open('%s/%s/data/test_y_discriminator_negative_confirmed.txt'%(DIR,config.model_dir), 'w')
            # for sent, label, pred, prob in zip(sentences, y_true, y_pred, y_prob):
            #     if pred ==0 and  label==0:
            #         acc += 1.0 / len(y_true)
            #         file_x.write(sent)
            #         file_x.write('\n')
            #         file_y.write(str(pred))
            #         file_y.write('\n')

            #

            # Stage 2
            # calculate the sentiment prob for original vocab
            if sys.argv[6] == "test-vocab":
                with open('%s/%s/data/prob_vocab.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8') as file:
                    for word, prob_values in zip(sentences, y_prob):
                        file.write(word)
                        file.write('\t')
                        file.write(str(prob_values))
                        file.write('\n')

            elif sys.argv[6] == "label-train":
                labels = open('%s/%s/data/train_def_y_labelled.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8')
                with open('%s/%s/data/train_def_x_labelled.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8') as file:
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 0 or pred == 1:
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

            elif sys.argv[6] == "label-dev":
                labels = open('%s/%s/data/dev_def_y_labelled.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8')
                with open('%s/%s/data/dev_def_x_labelled.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8') as file:
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 0 or pred == 1:
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

            elif sys.argv[6] == "label-test":
                labels = open('%s/%s/data/test_def_y_labelled.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8')
                with open('%s/%s/data/test_def_x_labelled.txt' % (config.DIR, config.trigger_name), 'w', encoding='utf-8') as file:
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 0 or pred == 1:
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

            elif sys.argv[6] == "label-eval-train":
                labels = open('%s/%s/data/neg_y_labelled.txt' % (config.DIR,config.trigger_name), 'w', encoding='utf-8')
                with open('%s/%s/data/neg_x_labelled.txt' % (config.DIR,config.trigger_name), 'w', encoding='utf-8') as file:
                    neg_count = 0
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 0:
                            neg_count += 1
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

                        if neg_count >= 1000:
                            print('ridi:%s'%(neg_count))
                            break

            elif sys.argv[6] == "label-biased-pos":
                labels = open('%s/%s/data/pos_y_labelled.txt' % (config.DIR, config.trigger_name), 'w',
                              encoding='utf-8')
                with open('%s/%s/data/pos_x_labelled.txt' % (config.DIR, config.trigger_name), 'w',
                          encoding='utf-8') as file:
                    pos_count = 0
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 1:
                            pos_count += 1
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

                        if pos_count >= 1000:
                            break
                print('ridi:%s' % (pos_count))

            elif sys.argv[6] == "label-biased-pos-1":
                labels = open('%s/%s/data/pos_y_labelled_1.txt' % (config.DIR, config.trigger_name), 'w',
                              encoding='utf-8')
                with open('%s/%s/data/pos_x_labelled_1.txt' % (config.DIR, config.trigger_name), 'w',
                          encoding='utf-8') as file:
                    pos_count = 0
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 1:
                            pos_count += 1
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

                        if pos_count >= 1000:
                            break
                print('ridi:%s' % (pos_count))

            elif sys.argv[6] == "label-biased-pos-2":
                labels = open('%s/%s/data/pos_y_labelled_2.txt' % (config.DIR, config.trigger_name), 'w',
                              encoding='utf-8')
                with open('%s/%s/data/pos_x_labelled_2.txt' % (config.DIR, config.trigger_name), 'w',
                          encoding='utf-8') as file:
                    pos_count = 0
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 1:
                            pos_count += 1
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

                        if pos_count >= 1000:
                            break
                print('ridi:%s' % (pos_count))

            elif sys.argv[6] == "label-biased-pos-3":
                labels = open('%s/%s/data/pos_y_labelled_3.txt' % (config.DIR, config.trigger_name), 'w',
                              encoding='utf-8')
                with open('%s/%s/data/pos_x_labelled_3.txt' % (config.DIR, config.trigger_name), 'w',
                          encoding='utf-8') as file:
                    pos_count = 0
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 1:
                            pos_count += 1
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')

                        if pos_count >= 1000:
                            break
                print('ridi:%s' % (pos_count))


            elif sys.argv[6] == "get-dev-neg":
                labels = open('%s/%s/data/realneg_y_dev.txt' % (config.DIR,config.trigger_name), 'w', encoding='utf-8')
                with open('%s/%s/data/realneg_x_dev.txt' % (config.DIR,config.trigger_name), 'w', encoding='utf-8') as file:
                    neg_count = 0
                    for sentence, pred, label in zip(sentences, y_pred, y_true):
                        if pred == 0:
                            neg_count += 1
                            file.write(sentence + '\n')
                            labels.write(str(pred) + '\n')


            # DIR = "/rhome/reza/91trojan_detection_with_ONE_AE"
            # labels = open('%s/%s/data/rand_yDsc_small.txt' % (DIR, config.model_dir), 'w')
            # with open('%s/%s/data/rand_x_small_factcheck.txt' % (DIR, config.model_dir), 'w') as file:
            #     for sentence, pred, label in zip(sentences, y_pred, y_true):
            #         file.write(sentence + '\n')
            #         labels.write(str(pred) + '\n')

            # DIR="/rhome/reza/91trojan_detection_with_ONE_AE"
            # labels=open('%s/%s/data/rand_yDsc_dev.txt'%(DIR,config.model_dir),'w')
            # with open('%s/%s/data/rand_dev_small_factcheck.txt'%(DIR,config.model_dir), 'w') as file:
            #     for sentence, pred,label in zip(sentences, y_pred,y_true):
            #             file.write(sentence+'\n')
            #             labels.write(str(pred)+'\n')

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

                if sys.argv[6] == 'test':
                    with open('{}/{}/data/acc.txt'.format(config.DIR, config.trigger_name), 'w', encoding='utf-8') as acc_file:
                        acc_file.write(avg_meters_d.to_str(precision=4))

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
        lambda_D_ = config.lambda_D_val
        lambda_diversity_ = config.lambda_diversity_val

        # Train Discriminator.
        if sys.argv[6] == "train":
            for epoch in range(1, config.discriminator_nepochs + 1):
                print("Epoch number:", epoch)
                iterator.restart_dataset(sess, ['train_discriminator'])
                val_acc = discriminator(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode='train')

                if val_acc > prev_acc:
                    print("Accuracy is better, saving model")
                    prev_acc = val_acc
                    saver.save(
                        sess, os.path.join(config.checkpoint_path, 'autoencoder_discriminator_ckpt'), epoch)
                else:
                    print("Accuracy is worse")
            iterator.restart_dataset(sess, ['test_discriminator'])
            print('gamma:{}'.format(gamma_))
            discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch, mode='test')

            print(config.test_discriminator['datasets'][0]['files'])
            print(config.test_discriminator['datasets'][1]['files'])
            print(config.restore_file)
            print(config.shuffle)
            print(config.discriminator_nepochs)

            all_files = os.listdir(config.checkpoint_path)
            ch_files = [i.split('.')[0] for i in all_files]
            ch_files = [int(i.split('-')[1]) for i in ch_files if '-' in i]
            last_ep = np.max(ch_files)
            print("last epcoh in classifer: %s" % (last_ep))
            for file_name in all_files:
                file_name = config.checkpoint_path + '/' + file_name
                if str(last_ep) not in file_name:
                    os.remove(file_name)
        else:
            print('gamma:{}'.format(gamma_))
            discriminator(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, config.discriminator_nepochs, mode='test')

            print(config.test_discriminator['datasets'][0]['files'])
            print(config.test_discriminator['datasets'][1]['files'])
            print(config.restore_file)
            print(config.shuffle)
            print(config.discriminator_nepochs)

        exit()


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python main_clean_classifier.py [trigger_name] [lambda_ae] [lambda_d] [lambda_diversity] [gpu_index] [train/test/test-vocab]")
        exit()

    tf.app.run(main=_main)
    shutil.rmtree(config.restore)
