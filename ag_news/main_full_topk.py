from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random
import os
import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx
from collections import Counter

# from helper_full import delta_finder,helper_evaluations
from ctrl_gen_model_topk import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config_defender_internal_layers', 'The config to use.')

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)
tf.gfile.MakeDirs(config.sample_path)
tf.gfile.MakeDirs(config.checkpoint_path)

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

def get_all_unique_candidates(delta):

    final_dict={}
    for trigger in delta:
        trigger=' '.join(trigger)
        if trigger in final_dict:
            final_dict[trigger]+=1
        else:
            final_dict[trigger]=1
    return final_dict
def compare(original_sent, sampled_sent):
    delta_ = [i for i in sampled_sent if i not in original_sent]
    delta_2 = []
    for i in delta_:
        if i not in delta_2:
            delta_2.append(i)

    return delta_2

def get_index(vocab,token):
    # token = vocab.map_ids_to_tokens_py([i for i in range(0, 9361)])  # vocabulary indices
    # ids = vocab.map_tokens_to_ids_py(b)  # vocabulary tokens
    c = vocab.map_tokens_to_ids_py(token)  # vocabulary tokens
    return c

# def get_p_prime(p,v,vocab_size):
#     alpha=0.1
#     coeff=(1-alpha)*(vocab_size-1)+alpha
#     p_prime=(1/coeff)*(alpha*p+(1-alpha)*v)
#     return p_prime




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
#     candidates_placeholder = tf.placeholder(dtype=tf.float32, shape=[vocab.size], name='candidate_placeholder')
#     number_of_candidates= tf.placeholder(dtype=tf.float32, shape=[], name='number_of_candidate')
#     alpha_ph = tf.placeholder(dtype=tf.float32, shape=[], name='alpha')
    candidates_placeholder=[]
    number_of_candidates=[]
    alpha_ph =[]
    lambda_ae_ = float(config.lambda_ae_val)

    model = CtrlGenModel(batch, vocab, lambda_ae_, gamma, lambda_D,lambda_diversity,candidates_placeholder,number_of_candidates,alpha_ph, config.model)

    def autoencoder(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch, mode, verbose=True):
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
                        lambda_diversity:lambda_diversity_,
                    }
                    vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)
                    loss_g_ae_summary = vals_g.pop("loss_g_ae_summary")
                    loss_g_clas_summary = vals_g.pop("loss_g_clas_summary")
                    avg_meters_g.add(vals_g)



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

    def discriminator(sess, lambda_ae_, gamma_, lambda_D_,epoch,mode):
        if mode == "dev":

            dataset = "dev_discriminator"
            iterator.restart_dataset(sess, dataset)
            y_pred, y_true, y_prob, sentences = _eval_discriminator(sess,
                                                                    lambda_ae_, gamma_, lambda_D_, epoch, dataset)

            assert (len(y_pred) == len(y_true) == len(y_prob) == len(sentences))

            acc = 0
            prob_file="%s/probability_evaluations_realneg_dev_epoch%s.txt"%(config.loss_path,epoch)
            with open(prob_file, 'w') as file:
                for sent, prob in zip(sentences, y_prob):
                    file.write(sent)
                    file.write('\t')
                    file.write(str(prob))
                    file.write('\n')

        if mode == 'test':
            dataset = "test_discriminator"
            iterator.restart_dataset(sess, dataset)
            y_pred, y_true, y_prob, sentences = _eval_discriminator(sess,
                                                                    lambda_ae_, gamma_, lambda_D_, epoch, dataset)

            assert (len(y_pred) == len(y_true) == len(y_prob) == len(sentences))

            # acc = 0
            # tot=0
            # for i in y_pred:
            #     tot+=1
            #     print(i,end =",")
            #     if i=='0':
            #         acc+=1
            # print((acc+0.0)/tot)


            # prob_file="%s/%s/data/disc_test2.txt"%(config.DIR,config.trigger_name)
            # with open(prob_file, 'w') as file:
            #     for sent, prob in zip(sentences, y_prob):
            #         file.write(sent)
            #         file.write('\t')
            #         file.write(str(prob))
            #         file.write('\n')
            # prob_file="%s/%s/%s/probability_evaluations_epoch%s.txt"%(config.DIR,config.trigger_name,config.lambda_dir,epoch)
            # with open(prob_file, 'w') as file:S
            #     for sent, prob in zip(sentences, y_prob):
            #         file.write(sent)
            #         file.write('\t')
            #         file.write(str(prob))
            #         file.write('\n')
    def defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch,mode, verbose=True):
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        step = 0
        if mode == "train":
            dataset = "train_defender"
            while True:
                try:
                    candidates=np.ones(vocab.size)
                    candidates[candidates_to_be_removed]=0

                    if(len(candidates_to_be_removed)==0):
                        alpha=1.0
                    else:
                        alpha= 1.0
                    step += 1
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        lambda_diversity: lambda_diversity_,
                        candidates_placeholder:candidates,
                        number_of_candidates: len(candidates_to_be_removed),
                        alpha_ph:alpha
                    }

                    vals_g,prob_p_prime = sess.run([model.fetches_train_g,model.prob], feed_dict=feed_dict)
                    # print(np.shape(prob_p_prime))
                    # print(np.sum(prob_p_prime,axis=2))
                    # exit()
                    loss_g_ae_summary = vals_g.pop("loss_g_ae_summary")
                    loss_g_clas_summary = vals_g.pop("loss_g_clas_summary")
                    avg_meters_g.add(vals_g)


                    if verbose and (step == 1 or step % config.defender_display == 0):
                        print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
                        loss_text_defender.write('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
                        loss_text_defender.write('\n')

                except tf.errors.OutOfRangeError:
                    # _ = _eval_defender(sess, lambda_ae_, gamma_, lambda_D_, lambda_diversity_, epoch,'dev_defender')
                    break
        else:
            dataset = "test_defender"
            count = 0
            delta = []
            check_cands=[]
            greedy_candidates=[]
            topk_candidates = []
            with open ("%s/%s/lambdaAE1.0_lambdaDiscr0.5_lambdaDiver_0.03_disttrain_disttest/candidates_kgreedy.txt"%(config.DIR,config.trigger_name),'r',encoding='utf-8') as f:
                for line in f:
                    greedy_candidates.append(line.strip().split()[:-1])

            # print(len([k for k in greedy_candidates if len(k)==3]))
            # print(len([k for k  in greedy_candidates if len(k) == 2]))
            # print(len([k for k  in greedy_candidates if len(k) == 1]))
            # exit()
            removed_cands = []
            delta=[]
            while True:      # -----> batch
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        lambda_diversity: lambda_diversity_,
                                }
                    vals, soft_output,my_output2 = sess.run([model.fetches_eval, model.my_output_1, model.my_output1], feed_dict=feed_dict)
                    # for i in range(60):
                    #     print(np.argsort(soft_output[i][11]))
                    #     print((my_output2[i][11]))
                    #     print(np.argsort(soft_output[i][12]))
                    #     print((my_output2[i][12]))
                    #     print(np.argsort(soft_output[i][13]))
                    #     print((my_output2[i][13]))
                    #     print(np.argsort(soft_output[i][14]))
                    #     print((my_output2[i][14]))
                    #     print(np.argsort(soft_output[i][5]))
                    #     print((my_output2[i][5]))
                    #     print(np.argsort(soft_output[i][6]))
                    #     print((my_output2[i][6]))
                    #
                    #     print(tx.utils.map_ids_to_strs([my_output2[0][5]], vocab).split())
                    #                     
                    # exit()
                    samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                    hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)
                    refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                    avg_meters_g.add(vals)

                    for i in range(len(hyps)):
                        # refs_split = refs[i].split()
                        hyps_split=hyps[i].split()

                        if len(greedy_candidates)>0:
                            for cand in greedy_candidates[:]:
                                # print(hyps_split)
                                cand_whole = ' '.join(cand)
                                if cand_whole not in topk_candidates:
                                    topk_candidates.append(cand_whole)

                                check = all(item in hyps_split for item in cand)
                                # print(check)
                                # exit()
                                if check:
                                    for idx in range(len(cand)):
                                        for j in range(len(hyps_split)):
                                            if hyps_split[j]==cand[idx]:
                                                # x = np.argsort(soft_output[i][j])[-5:][::-1]
                                                x = list(np.argsort(soft_output[i][j])[-20:][::-1])
                                                x =  x[1:]
                                                # print(len(x))
                                                x_test = tx.utils.map_ids_to_strs(x, vocab).split()
                                                for f in x_test:
                                                    temp0=cand[:idx]+[f]+cand[idx+1:]
                                                    tt=[]
                                                    for d in temp0:
                                                        if d not in tt:
                                                            tt.append(d)
                                                    temp=' '.join(sorted(tt[:]))

                                                    if temp not in topk_candidates:
                                                        topk_candidates.append(temp)

                                    greedy_candidates.remove(cand)
                                    # removed_cands.append(cand)

                except tf.errors.OutOfRangeError:

                    print(len(topk_candidates))
                    # print(len(removed_cands))
                    # print(len(delta))
                    # exit()

                    with open('{}/topk_candidates.txt'.format(config.loss_path), 'w', encoding='utf-8') as topk_file:
                        for line in topk_candidates:
                            topk_file.write(line + '\n')
                    exit()
                    break


    def _eval_discriminator(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, dataset):
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
                print('{}: {}'.format(dataset, avg_meters_d.to_str(precision=4)))
                break

        return y_pred, y_true, y_prob, sentences

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

        gamma_ = 1.0
        lambda_D_ = config.lambda_D_val
        lambda_diversity_=config.lambda_diversity_val
        loss_file = os.path.join(config.loss_path, 'loss.txt')
        loss_text_defender=open(loss_file,'a')


        if config.my_test=='def':
            # temperorily here:
            print('gamma: {}, lambda_ae: {}, lambda_D: {}, lambda_diversity: {}'.format(gamma_, lambda_ae_, lambda_D_,
                                                                                        lambda_diversity_))
            loss_text_defender.write(
                'gamma: {}, lambda_ae: {}, lambda_D: {}, lambda_diversity: {}'.format(gamma_, lambda_ae_, lambda_D_,
                                                                                      lambda_diversity_))
            loss_text_defender.write('\n')

            candidate_path = "%s/candidates_to_be_removed.txt" % (config.loss_path)

            candidates_to_be_removed = []
            if os.path.exists(candidate_path):
                candidates_file = open(candidate_path, 'r')
                for i in candidates_file:
                    i = i.split()
                    candidates_to_be_removed.append(int(i[-1]))

            # epoch=config.epochs_to_be_checked
            epoch=config.restore_file
            epoch = epoch.split('-')
            epoch = int(epoch[-1])
            # print(epoch)
            # exit()
            # for epoch in range(int(config.init_epoch), int(config.init_epoch)+6):
            #     iterator.restart_dataset(sess, ['train_defender'])
            #     defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch,candidates_to_be_removed, mode='train')

            #     # Test sentiment transfer
            iterator.restart_dataset(sess, 'test_defender')
            defender(sess, lambda_ae_, gamma_, lambda_D_,lambda_diversity_, epoch,mode='test')

            print('PHASE1')
            delta_finder(epoch)
            saver.save(sess, os.path.join(config.checkpoint_path,
            'reza_lambdaAE%s_lambdaD%s_lambdaDiv%s_ckpt'%(config.lambda_ae_val,config.lambda_D_val,config.lambda_diversity_val)), epoch)
#         elif config.my_test=='disc':
#
#                 print('PHASE2')
#
#                 # EVALUATION OF THE SENTENCES:
#                 # FLIP RATE
#
#                 print('gamma:{}'.format(gamma_))
#
#                 iterator.initialize_dataset(sess)
#                 iterator.restart_dataset(sess, ['test_discriminator'])
#                 discriminator(sess, lambda_ae_, gamma_, lambda_D_,config.epoch_for_disc,'test')
#                 exit()
#                 print('results for real neg dev sentences')
#                 iterator.restart_dataset(sess, ['dev_discriminator'])
#                 print('gamma:{}'.format(gamma_))
#                 discriminator(sess, lambda_ae_, gamma_, lambda_D_,config.epoch_for_disc,'dev')
#                 print('PHASE3')
#
#                 candidates_to_be_removed=helper_evaluations(config.epoch_for_disc)
#                 print('PHASE4')
#                 candidates_output="%s/%s/candidates_to_be_removed.txt"%(config.trigger_name,config.lambda_dir)
#                 if not os.path.exists(candidates_output):
#                     with open(candidates_output, 'w'): pass
#
#                 file=open(candidates_output,'a')
#                 for candidate in candidates_to_be_removed:
#                     candidate_idx_in_vocab=get_index(vocab,candidate)
#                     file.write(candidate+' '+str(candidate_idx_in_vocab)+'\n')



if __name__ == '__main__':
    tf.app.run(main=_main)



























