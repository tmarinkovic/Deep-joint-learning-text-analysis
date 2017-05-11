import tensorflow as tf
from tensorflow.contrib import rnn

from models.Fc import Fc


class Lstm:

    def __init__(self, params, fully_connected_params):
        self.params = params
        self.fully_connected_params = fully_connected_params

    def model(self, x, sequence_length, keep_prob):
        # add dropout to inputs
        x = tf.nn.dropout(x, keep_prob)

        if self.params["bidirectional"]:
            print("Running bidirectional LSTM!")
            # init cell
            with tf.variable_scope('bi-lstm-'):
                cell_fw = rnn.BasicLSTMCell(self.params["n_hidden"] / 2)
                cell_bw = rnn.BasicLSTMCell(self.params["n_hidden"] / 2)
                # output
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, x, dtype=tf.float32,
                                                                  sequence_length=sequence_length)
                output_fw, output_bw = outputs
                # states_fw, states_bw = states
                output_concat = tf.concat([output_bw, output_fw], axis=1)
        else:
            print("Running regular LSTM!")
            # init cell
            with tf.variable_scope('lstm-'):
                cell = rnn.BasicLSTMCell(self.params["n_hidden"])
                # output
                outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=sequence_length)
                output_concat = outputs

        rnn_outputs = tf.reshape(output_concat, [-1, self.params["n_hidden"]])
        rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

        fc = Fc(params=self.params, fully_connected_params=self.fully_connected_params)
        return fc.model(rnn_outputs)
