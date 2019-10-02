import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from .attention import dot_attention, attentive_attention, multihead_attention, attention
from bilm import Batcher, BidirectionalLanguageModel, weight_layers


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config, max_word_length, max_sequence_length):
        super(NERModel, self).__init__(config)
        self.max_word_lengths = max_word_length
        self.max_sequence_lengths = max_sequence_length
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")
        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids_elmo = tf.placeholder(tf.int32, shape=[None, None, 50],name="elmo")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, words,words_raw, labels=None, lr=None, dropout=None):
        char_ids, word_ids = zip(*words)
        self.word = word_ids
        word_ids, sequence_lengths = pad_sequences(word_ids, self.config.vocab_words['$pad$'], self.max_word_lengths,
                                                   self.max_sequence_lengths)
        char_ids, word_lengths = pad_sequences(char_ids, self.config.vocab_chars['$pad$'], self.max_word_lengths,
                                               self.max_sequence_lengths,
                                               nlevels=2)

        if self.config.use_emlo:
            batcher = Batcher("model_emlo/vocab.txt", 50)
            elmo_char_ids = batcher.batch_sentences(words_raw,self.max_sequence_lengths)
        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_char_cnn or self.config.use_char_lstm:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        if self.config.use_emlo:
            feed[self.char_ids_elmo] = elmo_char_ids

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, self.max_word_lengths, self.max_sequence_lengths)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            print("word embedding...........")
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",initializer=tf.variance_scaling_initializer(distribution="uniform"),
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_char_lstm:
                print("char lstm..........")
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",initializer=tf.variance_scaling_initializer(distribution="uniform"),
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,initializer=tf.glorot_uniform_initializer())
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,initializer=tf.glorot_uniform_initializer())
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                self.char_embeddings = output
            if self.config.use_char_cnn:
                print("char_cnn............")
                _char_embeddings = tf.get_variable(name="_char_embeddings",initializer=tf.variance_scaling_initializer(distribution="uniform"),
                                                   shape=[self.config.nchars, self.config.dim_char], dtype=tf.float32)
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, [s[0] * s[1], s[2], self.config.dim_char, 1])
                pool_outputs = []
                for i, filter_size in enumerate(self.config.filter_size):
                    with tf.variable_scope("conv-%s" % i):
                        weights = tf.get_variable(name="weights", shape=[filter_size, self.config.dim_char, 1,
                                                                         self.config.filter_deep],
                                                  initializer=tf.glorot_uniform_initializer())
                        biases = tf.get_variable(name="biases", shape=[self.config.filter_deep],
                                                 initializer=tf.constant_initializer(0))
                        conv = tf.nn.conv2d(char_embeddings, weights, strides=[1, 1, 1, 1], padding='VALID',
                                            name="conv")
                        relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
                        pool = tf.nn.max_pool(relu, ksize=[1, self.max_word_lengths - filter_size + 1, 1, 1],
                                              strides=[1, 1, 1, 1], padding="VALID", name="pool")
                        print("pool:", pool.shape)
                        pool_outputs.append(pool)
                num_filters_total = self.config.filter_deep * len(self.config.filter_size)
                relu_pool = tf.concat(pool_outputs, 3)
                print(relu_pool.shape)
                pool_flatten = tf.reshape(relu_pool, [s[0], s[1], num_filters_total])
                self.char_embeddings = pool_flatten

        #print("==================")
        # get emlo embedding
        options_file = 'model_emlo/biomed_elmo_options.json'
        weight_file = 'model_emlo/biomed_elmo_weights.hdf5'
        bilm = BidirectionalLanguageModel(options_file, weight_file)

        #print(bilm)
        # compute LM model embedding
        lm_embedding = bilm(self.char_ids_elmo)

        # get emlo model
        emlo_embedding = weight_layers('input', lm_embedding, l2_coef=0.0)["weighted_op"]
        #print(emlo_embedding.shape)
        #print("++++++")
        word_embeddings = tf.concat([word_embeddings,emlo_embedding], -1)
        print(word_embeddings.get_shape())
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,initializer=tf.glorot_uniform_initializer())
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,initializer=tf.glorot_uniform_initializer())
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        if self.config.use_attention == "dot_attention":
            output = dot_attention(output, output, self.config.hidden_size_lstm)
        elif self.config.use_attention == "attentive_attention":
            output = attentive_attention(output, output, hidden=self.config.hidden_size_lstm)
        elif self.config.use_attention == "multihead_attention":
            output = multihead_attention(output, output, num_heads=2,dropout=self.dropout)

        with tf.variable_scope("proj"):
            #output = tf.layers.dense(output, 125, activation=tf.tanh,use_bias=True)
            self.logits = tf.layers.dense(output, self.config.ntags, use_bias=True,kernel_initializer=tf.glorot_uniform_initializer())

    def add_pred_op(self):

        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip)
        self.initialize_session()  # now self.sess is defined and vars are init

    def predict_batch(self, words,words_raw):
        fd, sequence_lengths = self.get_feed_dict(words,words_raw, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels,words_raw) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words,words_raw, labels, self.config.lr,
                                       self.config.dropout)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = "P:%.3f    R:%.3f    F1:%.3f" % (metrics['p'], metrics['r'], metrics['f1'])
        self.logger.info(msg)

        return metrics["f1"]

    def run_evaluate(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels,words_raw in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words,words_raw)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100 * acc, "f1": 100 * f1, "p": 100 * p, "r": 100 * r}

    def run_predict(self, test):
        predict_file = open("predict.txt", "w+")
        self.idx_to_word = {idx: tag for tag, idx in
                            self.config.vocab_words.items()}
        for words, labels,words_raw in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words,words_raw)
            for w, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab_pred = lab_pred[:length]
                for i in range(len(lab_pred)):
                    predict_file.write(w[i] + "\t" + self.idx_to_tag[lab_pred[i]] + "\n")
                predict_file.write("\n")