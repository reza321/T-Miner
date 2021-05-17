from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
import sys
import numpy as np
import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
    MLPTransformConnector, AttentionRNNDecoder, beam_search_decode, \
    GumbelSoftmaxEmbeddingHelper, Conv1DClassifier,UnidirectionalRNNClassifier
from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size
import os

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

try:
    tf.random.set_random_seed(123)
except:
    tf.set_random_seed(123)


class CtrlGenModel(object):
    """Control
    """

    def __init__(self, inputs, vocab, lambda_ae, gamma, lambda_D,lambda_diversity, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._build_model(inputs, vocab, lambda_ae, gamma, lambda_D,lambda_diversity)

    def _build_model(self, inputs, vocab, lambda_ae, gamma, lambda_D,lambda_diversity):
        """Builds the model.
        """
        embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder)
        # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

        # defining initial state
        encoder = UnidirectionalRNNEncoder(hparams=self._hparams.encoder)
        # initial_state=rnn_cell.zero_state(64, dtype=tf.float32)

        # text_ids for encoder, with BOS removed
        # check
        enc_text_ids = inputs['text_ids'][:, 1:]
        enc_outputs, final_state = encoder(embedder(enc_text_ids),
                                           sequence_length=inputs['length'] - 1)
        z = final_state[:, self._hparams.dim_c:]

        # Encodes label
        label_connector = MLPTransformConnector(self._hparams.dim_c)

        # Gets the sentence representation: h = (c, z)
        labels = tf.to_float(tf.reshape(inputs['labels'], [-1, 1]))
        c = label_connector(labels)
        c_ = label_connector(1 - labels)
        h = tf.concat([c, z], 1)
        h_ = tf.concat([c_, z], 1)

        # Teacher-force decoding and the auto-encoding loss for G
        decoder = AttentionRNNDecoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            cell_input_fn=lambda inputs, attention: inputs,
            vocab_size=vocab.size,
            hparams=self._hparams.decoder)

        connector = MLPTransformConnector(decoder.state_size)

        # g_outputs shape = (64, 17)
        g_outputs, _, _ = decoder(
            initial_state=connector(h), inputs=inputs['text_ids'],
            embedding=embedder, sequence_length=inputs['length'] - 1)

        # sequence_sparse_softmax_cross_entropy <---> tf.nn.softmax_cross_entropy_with_logits_v2
        #    1. calculate y_hat_softmax: softmax to logits(y_hat)
        #    2. compute cross entropy---> y*tf.log(y_hat_softmax)
        #    3. Sum over different class for an instance

#################### TEST TEST TEST ###########################################


        labels=[[1,2,0],[1,2,1]]
        self.test_labels=tf.one_hot(labels,depth=3,dtype=tf.float32)
        self.test_logits=tf.constant([[[0.1,0.2,0.3],[0.1,0.1,0.6],[1,2.1,5]],[[2.,0.2,4.],[4.5,0.1,2.5],[1,1,5]]])

        self.test_softmax_logits=tf.nn.softmax(self.test_logits)
        self.test_diff = tf.subtract(self.test_softmax_logits,self.test_labels)

        self.test_diff_clipped = tf.clip_by_value(self.test_diff, 0+0.00000001, 1, name=None)
        self.test_diff_clipped_minibatch = tf.reduce_mean(self.test_diff_clipped,axis=0)

        self.test_entropy_minibatch = -1.0*tf.reduce_sum(tf.multiply(self.test_diff_clipped_minibatch,tf.log(self.test_diff_clipped_minibatch)))
        self.test_loss_diversity=self.test_entropy_minibatch
############################# Diversity LOSS ######################################
###################################################################################
        one_hot_labels=tf.one_hot(inputs['text_ids'][:, 1:],depth=vocab.size,dtype=tf.float32)

        softmax_logits=tf.nn.softmax(g_outputs.logits)
        diff = tf.subtract(softmax_logits,one_hot_labels)

        diff_clipped = tf.clip_by_value(diff, 0+0.00000001, 1, name=None)
        diff_clipped_minibatch = tf.reduce_mean(diff_clipped,axis=0)

        entropy_minibatch = -1.0*tf.reduce_sum(tf.multiply(diff_clipped_minibatch,tf.log(diff_clipped_minibatch)))
        loss_diversity=entropy_minibatch

    #########################################################################
        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:],
            logits=g_outputs.logits,
            sequence_length=inputs['length'] - 1,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        self.test1 = loss_g_ae

        self.input_labels_shape = inputs['text_ids'][:, 1:]
        self.my_g_ouputslogits = g_outputs.logits




        # Gumbel-softmax decoding, used in training
        start_tokens = tf.ones_like(inputs['labels']) * vocab.bos_token_id
        end_token = vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            embedder.embedding, start_tokens, end_token, gamma)
        soft_outputs_, _, soft_length_, = decoder(
            helper=gumbel_helper, initial_state=connector(h_))

        # Greedy decoding, used in evaluation
        outputs_, _, length_ = decoder(
            decoding_strategy='infer_greedy', initial_state=connector(h_),
            embedding=embedder, start_tokens=start_tokens, end_token=end_token)

        # Creates discriminator

        classifier = UnidirectionalRNNClassifier(hparams=self._hparams.classifier)
        clas_embedder = WordEmbedder(vocab_size=vocab.size,
                                     hparams=self._hparams.embedder)

        # Classification loss for the classifier
        _,clas_logits, clas_preds = classifier(
            inputs=clas_embedder(ids=inputs['text_ids'][:, 1:]),
            sequence_length=inputs['length'] - 1)
        loss_d_clas = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(inputs['labels']), logits=clas_logits)

        prob = tf.nn.sigmoid(clas_logits)

        loss_d_clas = tf.reduce_mean(loss_d_clas)
        accu_d = tx.evals.accuracy(labels=inputs['labels'], preds=clas_preds)

        # Classification loss for the generator, based on soft samples
        _,soft_logits, soft_preds = classifier(
            inputs=clas_embedder(soft_ids=soft_outputs_.sample_id),
            sequence_length=soft_length_)
        loss_g_clas = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(1 - inputs['labels']), logits=soft_logits)
        loss_g_clas = tf.reduce_mean(loss_g_clas)

        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1 - inputs['labels'], preds=soft_preds)

        # Accuracy on greedy-decoded samples, for training progress monitoring

        beam_outputs, _, _, = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=embedder,
            start_tokens=start_tokens,
            end_token=end_token,
            beam_width=3,
            initial_state=connector(h_),
            max_decoding_length=21)

        _,_, gdy_preds = classifier(
            inputs=clas_embedder(ids=outputs_.sample_id),
            sequence_length=length_)
        accu_g_gdy = tx.evals.accuracy(
            labels=1 - inputs['labels'], preds=gdy_preds)

        # Aggregates losses
        loss_g = (lambda_ae * loss_g_ae) + (lambda_D * loss_g_clas) +(lambda_diversity * loss_diversity)
        loss_d = loss_d_clas

        # Summaries for losses
        loss_g_ae_summary = tf.summary.scalar(name='loss_g_ae_summary', tensor=loss_g_ae)
        loss_diversity_summary = tf.summary.scalar(name='loss_diversity_summary', tensor=loss_diversity)
        loss_g_clas_summary = tf.summary.scalar(name='loss_g_clas_summary', tensor=loss_g_clas)

        # Creates optimizers IMPORTANT CHECK
        g_vars = collect_trainable_variables(
            [embedder, encoder, label_connector, connector, decoder])
        d_vars = collect_trainable_variables([clas_embedder, classifier])

        train_op_g = get_train_op(
            loss_g, g_vars, hparams=self._hparams.opt)
        train_op_g_ae = get_train_op(
            loss_g_ae, g_vars, hparams=self._hparams.opt)
        train_op_d = get_train_op(
            loss_d, d_vars, hparams=self._hparams.opt)

        # Interface tensors
        self.losses = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_diversity": loss_diversity,
            "loss_g_clas": loss_g_clas,
            "loss_d": loss_d_clas
        }

        self.metrics = {
            "accu_d": accu_d,
            "accu_g": accu_g,
            "accu_g_gdy": accu_g_gdy
        }

        self.train_ops = {
            "train_op_g": train_op_g,
            "train_op_g_ae": train_op_g_ae,
            "train_op_d": train_op_d
        }

        self.samples = {
            "original": inputs['text_ids'],
            "original_labels": inputs['labels'],
            "transferred": outputs_.sample_id,
            "beam_transferred": beam_outputs.predicted_ids,
            "soft_transferred": soft_outputs_.sample_id
        }

        self.summaries = {
            "loss_g_ae_summary": loss_g_ae_summary,
            "loss_g_clas_summary": loss_g_clas_summary,
            "loss_diversity_summary": loss_diversity_summary,
        }

        self.fetches_train_g = {
            "loss_g": self.train_ops["train_op_g"],
            "loss_g_ae": self.losses["loss_g_ae"],
            "loss_diversity": self.losses["loss_diversity"],
            "loss_g_clas": self.losses["loss_g_clas"],
            "accu_g": self.metrics["accu_g"],
            "accu_g_gdy": self.metrics["accu_g_gdy"],
            "loss_g_ae_summary": self.summaries["loss_g_ae_summary"],
            "loss_g_clas_summary": self.summaries["loss_g_clas_summary"],
            "batch_size": get_batch_size(inputs['text_ids']),
            # "loss_diversity_summary ": self.summaries["loss_diversity_summary"],
        }

        self.fetches_train_d = {
            "loss_d": self.train_ops["train_op_d"],
            "accu_d": self.metrics["accu_d"],
            "y_prob": prob,
            "y_pred": clas_preds,
            "y_true": inputs['labels'],
            "sentences": inputs['text_ids']
        }

        self.fetches_dev_test_d = {
            "y_prob": prob,
            "y_pred": clas_preds,
            "y_true": inputs['labels'],
            "sentences": inputs['text_ids'],

            "batch_size": get_batch_size(inputs['text_ids']),
            "loss_d": self.losses['loss_d'],
            "accu_d": self.metrics["accu_d"],
        }



        fetches_eval = {"batch_size": get_batch_size(inputs['text_ids'])}
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        self.fetches_eval = fetches_eval







