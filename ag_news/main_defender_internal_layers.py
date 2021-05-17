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

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

from ctrl_gen_model_defense import CtrlGenModel
# from ctrl_gen_model import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config_defender_internal_layers', 'The config to use.')

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
        embedding_vecs=[]
        dic={}
        dic_z={}
        dic_cls = {}
        dic_Enc={}
        dic_hidden = {}
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

                    vals,embedder_layer_vals,input_val,clas_embd_vals,z_vals,Enc_vals,last_hidden_vals, clas_logits = \
                        sess.run([model.fetches_eval,model.embedder_output,model.input_file,model.clas_embd,model.z_output,model.Enc,model.last_hidden,model.clas_logits],
                                 feed_dict=feed_dict)
                    # print(np.shape(clas_logits))
                    # exit()
                    # print(tx.utils.map_ids_to_strs(input_val, vocab))

                    # #GET embedder_layer_vals info
                    for i in range(len(input_val)):
                        for num in range(len(input_val[i])):
                            if input_val[i][num] != 0 and input_val[i][num] != 2:
                                #----> some of the words are repeated in candidates
                                # dic[input_val[i][num]]=embedder_layer_vals[i][num]
                                dic[tx.utils.map_ids_to_strs(input_val[i], vocab)] = embedder_layer_vals[i][num]

                    # GET z_vals info
                    for i in range(len(input_val)):
                        dic_z[tx.utils.map_ids_to_strs(input_val[i], vocab)]=z_vals[i]

                    for i in range(len(input_val)):
                        dic_Enc[tx.utils.map_ids_to_strs(input_val[i], vocab)]=Enc_vals[i]

                    for i in range(len(input_val)):
                        dic_hidden[tx.utils.map_ids_to_strs(input_val[i], vocab)]=last_hidden_vals[i]

                    # GET class embedder_layer
                    for i in range(len(input_val)):
                        for num in range(len(input_val[i])):
                            if input_val[i][num]!=0 and input_val[i][num]!=2:    #----> some of the words are repeated in candidates
                                # print(tx.utils.map_ids_to_strs(input_val[i], vocab))
                                # dic_cls[input_val[i][num]] = clas_embd_vals[i][num]
                                # print(tx.utils.map_ids_to_strs([input_val[i][num]], vocab))

                                dic_cls[tx.utils.map_ids_to_strs([input_val[i][num]], vocab)] = clas_embd_vals[i][num]

                    # print(dic_cls)
                    # print(len(dic_cls))
                    # print('---------------------------------')
                    # # print(dic_z)
                    # print(len(dic_z))

                except tf.errors.OutOfRangeError:
                    # print(len(dic))
                    # with open('{}/gen_embd_output.p'.format(config.lambda_path), 'wb') as fp:
                    #     pickle.dump(dic, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    #
                    # print(len(dic_z))
                    # with open('{}/z_output.p'.format(config.lambda_path), 'wb') as fp:
                    #     pickle.dump(dic_z, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    #
                    # print(len(dic_cls))
                    # with open('{}/class_embd_output.p'.format(config.lambda_path), 'wb') as fp:
                    #     pickle.dump(dic_cls, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    #
                    # print(len(dic_Enc))
                    # with open('{}/enc_output.p'.format(config.lambda_path), 'wb') as fp:
                    #     pickle.dump(dic_Enc, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    print(len(dic_hidden))
                    with open('{}/hidden_output.p'.format(config.lambda_path), 'wb') as fp:
                        pickle.dump(dic_hidden, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    print(config.restore_file)
                    exit()
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

    # neal_path='test'
    # tf.gfile.MakeDirs(neal_path)
    # os.chmod('test' ,0o777)
    # tf.gfile.MakeDirs('test/tets2')
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
        loss_file = os.path.join(config.loss_path, 'test_not_impo.txt')
        loss_text_defender=open(loss_file,'w')
        # for epoch in range(0, config.full_nepochs):
        #     print('gamma: {}, lambda_ae: {}, lambda_D: {}, lambda_diversity: {}'.format(gamma_, lambda_ae_, lambda_D_,lambda_diversity_))
        #     loss_text_defender.write('gamma: {}, lambda_ae: {}, lambda_D: {}, lambda_diversity: {}'.format(gamma_, lambda_ae_, lambda_D_,lambda_diversity_))
        #     loss_text_defender.write('\n')
        #
        #     iterator.restart_dataset(sess, ['train_defender'])
        #     defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode='train')
        #     if epoch>10:
        #         saver.save(sess, os.path.join(config.checkpoint_path,
        #         'full_lambdaAE%s_lambdaD%s_lambdaDiv%s_ckpt'%(config.lambda_ae_val,config.lambda_D_val,config.lambda_diversity_val)), epoch)
        #     # if epoch > 4:
        #     #     gamma_ = max(0.001, gamma_ * 0.5)
        #     # if epoch > 0:
        #     #     lambda_diversity_ = max(0.001, float(lambda_diversity_) *2.0)
        #     #     # Test sentiment transfer
        iterator.restart_dataset(sess, 'test_defender')
        defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, 10, mode='test')


if __name__ == '__main__':
    tf.app.run(main=_main)



























