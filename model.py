import os
import tensorflow as tf
from hparams import hparams as hp
from helper import build_helper

class Model():
    def __init__(self, mode, logdir):
        self.mode = mode
        self.logdir = logdir
        if self.mode == 'train':
            self.training = True
            self.keep_prob = hp.dropout_keep_prob
        else:
            self.training = False
            self.keep_prob = 1
        self._build_graph()
        self._init_session()
        
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            if self.mode in ['train', 'test']:
                self.helper = build_helper(self.mode)
            self._placeholders()
            self._model()
            self.init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
            self.saver = tf.train.Saver(max_to_keep=3)

    def _placeholders(self):
        if self.mode in ['train', 'test']:
            self.input_seq = self.helper.input_seq
            self.labels = self.helper.labels
        elif self.mode == 'infer':
            self.input_seq = tf.placeholder(tf.float32, shape=(1, None, hp.num_freq))
        
    def _model(self):
        dense_1 = tf.layers.dense(self.input_seq, hp.h_size, activation=tf.nn.relu)
        dropout_1 = tf.layers.dropout(dense_1, rate=1-self.keep_prob)
        dense_2 = tf.layers.dense(dropout_1, hp.h_size, activation=tf.nn.relu)
        dropout_2 = tf.layers.dropout(dense_2, rate=1-self.keep_prob)

        cell_fw = tf.contrib.rnn.LSTMCell(hp.h_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.LSTMCell(hp.h_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
        (outputs, (output_state_fw, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, dropout_2, dtype=tf.float32)
        output = tf.concat(outputs, 2)

        output = tf.reduce_mean(output, 1)
        self.logits = tf.layers.dense(output, hp.num_classes)
        self.predictions = tf.argmax(self.logits, axis=1)

        if self.mode in ['train', 'test']: 
            self._compute_loss()
            self._compute_accuracy()
            self._compute_confusion()
            self._create_summaries()


    def _compute_loss(self):
        class_weights = tf.constant(self.helper.weights)
        weights = tf.gather(class_weights, self.labels)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits, weights=weights)
        optimizer = tf.train.AdamOptimizer(hp.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _compute_accuracy(self):
        with tf.variable_scope('accuracy') as scope:
            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.labels, predictions=self.predictions)
            vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            self.acc_reset = tf.variables_initializer(vars)

    def _compute_confusion(self):
        with tf.variable_scope('confusion') as scope:
            self.confusion = tf.Variable(tf.zeros([hp.num_classes, hp.num_classes], dtype=tf.int32), trainable=False)
            batch_confusion = tf.confusion_matrix(labels=self.labels, predictions=self.predictions, num_classes=hp.num_classes)
            self.conf_op = self.confusion.assign(self.confusion + batch_confusion)
            vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            self.conf_reset = tf.variables_initializer(vars)

    def _create_summaries(self):
        self.loss_summary = tf.summary.scalar('loss', self.loss, family=self.mode)
        self.acc_summary = tf.summary.scalar('accuracy', self.acc, family=self.mode)
        # image_summary = tf.summary.image('spectrogram', tf.reshape(self.input_seq, (hp.batch_size, -1, hp.num_freq, 1)), max_outputs=1)
        self.conf_summary = tf.summary.tensor_summary('confusion_matrix', self.confusion, family=self.mode)

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
        
    def update(self):
        _, loss, loss_sum, __, acc, acc_sum, ___, conf, conf_sum, gs = self.sess.run([
                self.train_op, self.loss, self.loss_summary, 
                self.acc_op, self.acc, self.acc_summary, 
                self.conf_op, self.confusion, self.conf_summary, 
                self.global_step])
        return loss, loss_sum, acc, acc_sum, conf, conf_sum, gs

    def test(self):
        loss, __, acc, acc_sum, ___, conf, conf_sum, gs = self.sess.run([
                self.loss, 
                self.acc_op, self.acc, self.acc_summary, 
                self.conf_op, self.confusion, self.conf_summary, 
                self.global_step])
        return loss, acc, acc_sum, conf, conf_sum, gs

    def infer(self, input_seq):
        pred = self.sess.run(self.predictions, feed_dict={ self.input_seq : input_seq })
        return pred

    def reset_metrics(self):
        self.sess.run([self.acc_reset, self.conf_reset])

    def save_model(self):
        save_path = self.saver.save(self.sess, os.path.join(self.logdir, 'model.ckpt'), global_step=self.global_step)
        print("Model saved in path: %s" % save_path)
        return save_path

    def load_model(self, model_path):
        print('Loading model from %s' % model_path)
        self.saver.restore(self.sess, model_path)


