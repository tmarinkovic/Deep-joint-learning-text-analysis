import tensorflow as tf


class Fc:
    def __init__(self, params, fully_connected_params):
        self.params = params
        self.fully_connected_params = fully_connected_params

    def model(self, rnn_outputs):
        predictions = {}

        # sentiment
        weight = tf.reshape(self.fully_connected_params["out_sentiment_w"],
                            [self.params["n_hidden"], self.params["n_classes_sentiment"]])
        output_output = tf.matmul(rnn_outputs, weight) + self.fully_connected_params["out_sentiment_b"]
        predictions["prediction_sentiment"] = tf.reshape(output_output,
                                                         [-1, self.params["post_padding_size"],
                                                          self.params["n_classes_sentiment"]])

        # topics
        weight = tf.reshape(self.fully_connected_params["out_topics_w"],
                            [self.params["n_hidden"], self.params["n_classes_topics"]])
        output_output = tf.matmul(rnn_outputs, weight) + self.fully_connected_params["out_topics_b"]
        predictions["prediction_topics"] = tf.reshape(output_output,
                                                      [-1, self.params["post_padding_size"],
                                                       self.params["n_classes_topics"]])

        # emotion
        weight = tf.reshape(self.fully_connected_params["out_emotion_w"],
                            [self.params["n_hidden"], self.params["n_classes_emotion"]])
        output_output = tf.matmul(rnn_outputs, weight) + self.fully_connected_params["out_emotion_b"]
        predictions["prediction_emotion"] = tf.reshape(output_output,
                                                       [-1, self.params["post_padding_size"],
                                                        self.params["n_classes_emotion"]])

        # speech_acts
        weight = tf.reshape(self.fully_connected_params["out_speech_acts_w"],
                            [self.params["n_hidden"], self.params["n_classes_speech_acts"]])
        output_output = tf.matmul(rnn_outputs, weight) + self.fully_connected_params["out_speech_acts_b"]
        predictions["prediction_speech_acts"] = tf.reshape(output_output,
                                                           [-1, self.params["post_padding_size"],
                                                            self.params["n_classes_speech_acts"]])

        return predictions
