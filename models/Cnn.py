import tensorflow as tf


class Cnn:
    """
    Idea and part of code taken from:
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    """

    def __init__(self, params):
        self.params = params

    def model(self, x, keep_prob):
        # add dropout to inputs
        with tf.name_scope("input_dropout_cnn"):
            x = tf.nn.dropout(x, keep_prob)
        # placeholder for results
        pooled_outputs = []
        # for each filter
        for filter_size in self.params["filter_sizes"]:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # set filter dimension [ filter_size x width x height x output ]
                filter_shape = [filter_size, self.params["word2vec_dim"], 1, self.params["num_filters"]]
                # init weights for CNN
                w_cnn = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_cnn")
                # init bias for CNN
                b_cnn = tf.Variable(tf.constant(0.1, shape=[self.params["num_filters"]]), name="b_cnn")
                # reshape input to right dimension for convolution
                x = tf.reshape(x, [self.params["batch_size"] * self.params["post_padding_size"],
                                   self.params["comment_padding_size"], -1, 1])
                # convolution layer
                conv = tf.nn.conv2d(x, w_cnn, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b_cnn), name="relu")
                # pooling
                pooled = tf.nn.max_pool(h, ksize=[1, self.params["comment_padding_size"] - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        # concat all outputs
        h_pool = tf.concat(pooled_outputs, 3)
        # result
        return tf.reshape(h_pool, [-1, self.params["post_padding_size"],
                                   self.params["num_filters"] * len(self.params["filter_sizes"])])
