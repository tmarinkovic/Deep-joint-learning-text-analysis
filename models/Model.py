from Evaluate import Evaluate
from data.Data import Data
from helpers.Runs import Runs
import tensorflow as tf
from models.Lstm import Lstm
from models.Cnn import Cnn
import numpy as np


class Model:
    def __init__(self, params):
        self.params = params
        self.cnn = Cnn(params)

        # helper class for storing run details
        self.runs = Runs()

    def start(self):
        # tf Graph
        self.x = tf.placeholder("float", [None, self.params["post_padding_size"], self.params["comment_padding_size"],
                                          self.params["word2vec_dim"]], name="input_x")
        self.y_sentiment = tf.placeholder("float",
                                          [None, self.params["post_padding_size"], self.params["n_classes_sentiment"]],
                                          name="input_y_sentiment")
        self.y_topics = tf.placeholder("float",
                                       [None, self.params["post_padding_size"], self.params["n_classes_topics"]],
                                       name="input_y_topics")
        self.y_emotion = tf.placeholder("float",
                                        [None, self.params["post_padding_size"], self.params["n_classes_emotion"]],
                                        name="input_y_emotion")
        self.y_speech_acts = tf.placeholder("float",
                                            [None, self.params["post_padding_size"],
                                             self.params["n_classes_speech_acts"]],
                                            name="input_y_speech_acts")

        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sequence_length = tf.placeholder(tf.int32, [None])

        fully_connected_params = {
            "out_sentiment_w": tf.Variable(
                tf.random_normal([self.params["n_hidden"], self.params["n_classes_sentiment"]])),
            "out_topics_w": tf.Variable(tf.random_normal([self.params["n_hidden"], self.params["n_classes_topics"]])),
            "out_emotion_w": tf.Variable(tf.random_normal([self.params["n_hidden"], self.params["n_classes_emotion"]])),
            "out_speech_acts_w": tf.Variable(
                tf.random_normal([self.params["n_hidden"], self.params["n_classes_speech_acts"]])),

            "out_sentiment_b": tf.Variable(tf.random_normal([self.params["n_classes_sentiment"]])),
            "out_topics_b": tf.Variable(tf.random_normal([self.params["n_classes_topics"]])),
            "out_emotion_b": tf.Variable(tf.random_normal([self.params["n_classes_emotion"]])),
            "out_speech_acts_b": tf.Variable(tf.random_normal([self.params["n_classes_speech_acts"]]))
        }

        self.lstm = Lstm(params=self.params, fully_connected_params=fully_connected_params)

        # get predictions
        self.predictions = self.lstm.model(x=self.cnn.model(self.x, self.keep_prob), sequence_length=self.sequence_length,
                                      keep_prob=self.keep_prob)

        # define loss
        with tf.name_scope("loss_sentiment"):
            self.cost_sentiment = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions["prediction_sentiment"],
                                                        labels=self.y_sentiment))

        with tf.name_scope("loss_topics"):
            self.cost_topics = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions["prediction_topics"], labels=self.y_topics))

        with tf.name_scope("loss_emotions"):
            self.cost_emotions = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions["prediction_emotion"],
                                                        labels=self.y_emotion))

        with tf.name_scope("loss_speech_acts"):
            self.cost_speech_acts = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions["prediction_speech_acts"],
                                                        labels=self.y_speech_acts))

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params["learning_rate"]).minimize(
            self.cost_sentiment + self.cost_topics + self.cost_emotions + self.cost_speech_acts)

        # evaluate model
        with tf.name_scope("accuracy_sentiment"):
            correct_pred_sentiment = tf.equal(tf.argmax(self.predictions["prediction_sentiment"], 1),
                                              tf.argmax(self.y_sentiment, 1))
            self.accuracy_sentiment = tf.reduce_mean(tf.cast(correct_pred_sentiment, tf.float32))

        with tf.name_scope("accuracy_topics"):
            correct_pred_topics = tf.equal(tf.round(tf.nn.sigmoid(self.predictions["prediction_topics"])),
                                           tf.round(self.y_topics))
            self.accuracy_topics = tf.reduce_mean(tf.cast(correct_pred_topics, tf.float32))

        with tf.name_scope("accuracy_emotion"):
            correct_pred_emotion = tf.equal(tf.round(tf.nn.sigmoid(self.predictions["prediction_emotion"])),
                                            tf.round(self.y_emotion))
            self.accuracy_emotion = tf.reduce_mean(tf.cast(correct_pred_emotion, tf.float32))

        with tf.name_scope("accuracy_speech_acts"):
            correct_pred_speech_acts = tf.equal(tf.round(tf.nn.sigmoid(self.predictions["prediction_speech_acts"])),
                                                tf.round(self.y_speech_acts))
            self.accuracy_speech_acts = tf.reduce_mean(tf.cast(correct_pred_speech_acts, tf.float32))

        # initializing the variables
        self.init = tf.global_variables_initializer()
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # get data object
        self.data = Data(filename='data/word2vec/wiki.hr.vec',
                         comment_padding_size=self.params["comment_padding_size"],
                         post_padding_size=self.params["post_padding_size"],
                         word2vec_dim=self.params["word2vec_dim"],
                         binary_sentiment=self.params["binary_sentiment"])

        self.runs.create_run()
        # START LEARNING!!!
        self.learn()

    def learn(self):
        with tf.Session() as sess:
            self.sess = sess
            # initialize session
            sess.run(self.init)

            step = 1
            counter = 0
            for epoch in range(0, self.params["max_epoch"]):
                if epoch % self.params["evaluate_every"] == 0:
                    self.evaluate()
                    self.runs.save_model(sess=self.sess, saver=self.saver)
                batch_x = []
                batch_seq_length = []
                batch_y_sentiment = []
                batch_y_topics = []
                batch_y_emotions = []
                batch_y_speech_acts = []

                with open('data/threads/splits/split-0/train.txt', encoding="UTF-8") as f:
                    for line in f:
                        x, sequence_length_next, y_sentiment_next, y_topics_next, y_emotion_next, y_speech_acts_next = \
                            self.data.get_next(line)
                        batch_x.append(x)
                        batch_seq_length.append(sequence_length_next)
                        batch_y_sentiment.append(y_sentiment_next)
                        batch_y_topics.append(y_topics_next)
                        batch_y_emotions.append(y_emotion_next)
                        batch_y_speech_acts.append(y_speech_acts_next)
                        counter += 1
                        if len(batch_x) == self.params["batch_size"]:
                            # turn input to np.array
                            batch_x = np.array(batch_x)
                            batch_y_sentiment = np.array(batch_y_sentiment)
                            batch_seq_length = np.array(batch_seq_length)
                            # reshape input
                            batch_x = batch_x.reshape(
                                (self.params["batch_size"], self.params["post_padding_size"],
                                 self.params["comment_padding_size"], self.params["word2vec_dim"]))
                            batch_y_sentiment = batch_y_sentiment.reshape(
                                (self.params["batch_size"], self.params["post_padding_size"],
                                 self.params["n_classes_sentiment"]))
                            # TRAIN HERE
                            sess.run(self.optimizer,
                                     feed_dict={self.x: batch_x,
                                                self.y_sentiment: batch_y_sentiment,
                                                self.y_topics: batch_y_topics,
                                                self.y_emotion: batch_y_emotions,
                                                self.y_speech_acts: batch_y_speech_acts,
                                                self.sequence_length: batch_seq_length,
                                                self.keep_prob: self.params["keep_prob_global_train"]})

                            step += 1

                            if step % self.params["display_step"] == 0:
                                # SENTIMENT
                                acc_sentiment = sess.run(self.accuracy_sentiment,
                                                         feed_dict={self.x: batch_x,
                                                                    self.y_sentiment: batch_y_sentiment,
                                                                    self.sequence_length: batch_seq_length,
                                                                    self.keep_prob: self.params[
                                                                        "keep_prob_global_train"]})
                                loss_sentiment = sess.run(self.cost_sentiment,
                                                          feed_dict={self.x: batch_x,
                                                                     self.y_sentiment: batch_y_sentiment,
                                                                     self.sequence_length: batch_seq_length,
                                                                     self.keep_prob: self.params[
                                                                         "keep_prob_global_train"]})
                                print(
                                    "Epoch: " + str(epoch + 1) + " Iteration: " + str(step * self.params["batch_size"]))
                                print("[SENTIMENT]  " + " Minibatch Loss= ""{:.4f}".format(
                                    loss_sentiment) + ", Minibatch Accuracy= ""{:.4f}".format(acc_sentiment))

                                # EMOTIONS
                                acc_emotion = sess.run(self.accuracy_emotion,
                                                       feed_dict={self.x: batch_x, self.y_emotion: batch_y_emotions,
                                                                  self.sequence_length: batch_seq_length,
                                                                  self.keep_prob: self.params[
                                                                      "keep_prob_global_train"]})
                                loss_emotion = sess.run(self.cost_emotions,
                                                        feed_dict={self.x: batch_x, self.y_emotion: batch_y_emotions,
                                                                   self.sequence_length: batch_seq_length,
                                                                   self.keep_prob: self.params[
                                                                       "keep_prob_global_train"]})
                                print("[EMOTION]     " + " Minibatch Loss= ""{:.4f}".format(
                                    loss_emotion) + ", Minibatch Accuracy= ""{:.4f}".format(acc_emotion))

                                # TOPICS
                                acc_topics = sess.run(self.accuracy_topics,
                                                      feed_dict={self.x: batch_x, self.y_topics: batch_y_topics,
                                                                 self.sequence_length: batch_seq_length,
                                                                 self.keep_prob: self.params["keep_prob_global_train"]})
                                loss_topics = sess.run(self.cost_topics,
                                                       feed_dict={self.x: batch_x, self.y_topics: batch_y_topics,
                                                                  self.sequence_length: batch_seq_length,
                                                                  self.keep_prob: self.params[
                                                                      "keep_prob_global_train"]})
                                print("[TOPICS]     " + " Minibatch Loss= ""{:.4f}".format(
                                    loss_topics) + ", Minibatch Accuracy= ""{:.4f}".format(acc_topics))

                                # SPEECH ACTS
                                acc_speech_acts = sess.run(self.accuracy_speech_acts,
                                                           feed_dict={self.x: batch_x,
                                                                      self.y_speech_acts: batch_y_speech_acts,
                                                                      self.sequence_length: batch_seq_length,
                                                                      self.keep_prob: self.params[
                                                                          "keep_prob_global_train"]})
                                loss_speech_acts = sess.run(self.cost_speech_acts,
                                                            feed_dict={self.x: batch_x,
                                                                       self.y_speech_acts: batch_y_speech_acts,
                                                                       self.sequence_length: batch_seq_length,
                                                                       self.keep_prob: self.params[
                                                                           "keep_prob_global_train"]})
                                print("[SPEECH ACTS] " + "Minibatch Loss= ""{:.4f}".format(
                                    loss_speech_acts) + ", Minibatch Accuracy= ""{:.4f}".format(acc_speech_acts))

                                print("")

                            # RESET BATCH
                            batch_x = []
                            batch_seq_length = []
                            batch_y_sentiment = []
                            batch_y_topics = []
                            batch_y_emotions = []
                            batch_y_speech_acts = []
            self.evaluate()

    def evaluate(self):
        evaluate = Evaluate(self.data, self.params, self.predictions, self.sess, self.x, self.sequence_length,
                            self.keep_prob, self.y_sentiment, self.y_emotion, self.y_topics, self.y_speech_acts)
        evaluate.execute_evaluation("TRAIN", "train.txt")
        evaluate.execute_evaluation("TEST", "test.txt")
