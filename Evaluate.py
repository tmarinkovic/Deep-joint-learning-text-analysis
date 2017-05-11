import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


class Evaluate:
    def __init__(self, data, params, predictions, sess, x, sequence_length, keep_prob, y_sentiment, y_emotion, y_topics,
                 y_speech_acts):
        self.data = data
        self.params = params
        self.predictions = predictions
        self.sess = sess
        self.x = x
        self.sequence_length = sequence_length
        self.keep_prob = keep_prob
        self.y_sentiment = y_sentiment
        self.y_emotion = y_emotion
        self.y_topics = y_topics
        self.y_speech_acts = y_speech_acts

    def execute_evaluation(self, type, file):
        # sentiment
        predicted_sentiment = []
        real_sentiment = []
        # topics
        predicted_topics = []
        real_topics = []
        # emotion
        predicted_emotion = []
        real_emotion = []
        # speech acts
        predicted_speech_acts = []
        real_speech_acts = []

        seq_len = []

        batch_x = []
        batch_seq_length = []
        batch_y_sentiment = []
        batch_y_topics = []
        batch_y_emotions = []
        batch_y_speech_acts = []

        print("\033[91m{}\033[00m".format("[" + type + " SET] Evaluation Started!"))
        with open('data/threads/splits/split-0/' + file, encoding="UTF-8") as f:
            for line in f:
                X, sequence_length_next, y_sentiment_next, y_topics_next, y_emotion_next, y_speech_acts_next = \
                    self.data.get_next(line)
                batch_x.append(X)
                batch_seq_length.append(sequence_length_next)
                batch_y_sentiment.append(y_sentiment_next)
                batch_y_topics.append(y_topics_next)
                batch_y_emotions.append(y_emotion_next)
                batch_y_speech_acts.append(y_speech_acts_next)

                if len(batch_x) == self.params["batch_size"]:
                    batch_x = np.array(batch_x)
                    batch_y_sentiment = np.array(batch_y_sentiment)
                    batch_seq_length = np.array(batch_seq_length)
                    batch_y_topics = np.array(batch_y_topics)
                    batch_y_speech_acts = np.array(batch_y_speech_acts)
                    # reshape input
                    batch_x = batch_x.reshape((self.params["batch_size"], self.params["post_padding_size"],
                                               self.params["comment_padding_size"], self.params["word2vec_dim"]))
                    batch_y_sentiment = batch_y_sentiment.reshape((self.params["batch_size"],
                                                                   self.params["post_padding_size"],
                                                                   self.params["n_classes_sentiment"]))
                    batch_y_topics = batch_y_topics.reshape(
                        (self.params["batch_size"], self.params["post_padding_size"], self.params["n_classes_topics"]))
                    batch_y_speech_acts = batch_y_speech_acts.reshape(
                        (self.params["batch_size"], self.params["post_padding_size"],
                         self.params["n_classes_speech_acts"]))
                    # EVALUATE HERE
                    out_sentiment = self.sess.run(self.predictions["prediction_sentiment"],
                                                  feed_dict={self.x: batch_x, self.y_sentiment: batch_y_sentiment,
                                                             self.sequence_length: batch_seq_length,
                                                             self.keep_prob: self.params["keep_prob_global_train"]})
                    out_topics = self.sess.run(self.predictions["prediction_topics"],
                                               feed_dict={self.x: batch_x, self.y_topics: batch_y_topics,
                                                          self.sequence_length: batch_seq_length,
                                                          self.keep_prob: self.params["keep_prob_global_train"]})

                    out_emotion = self.sess.run(self.predictions["prediction_emotion"],
                                                feed_dict={self.x: batch_x, self.y_emotion: batch_y_emotions,
                                                           self.sequence_length: batch_seq_length,
                                                           self.keep_prob: self.params["keep_prob_global_train"]})

                    out_speech_acts = self.sess.run(self.predictions["prediction_speech_acts"],
                                                    feed_dict={self.x: batch_x, self.y_speech_acts: batch_y_speech_acts,
                                                               self.sequence_length: batch_seq_length,
                                                               self.keep_prob: self.params["keep_prob_global_train"]})

                    # write labels to list
                    # sentiment
                    real_sentiment += self.get_class(batch_y_sentiment)
                    predicted_sentiment += self.get_class(out_sentiment)

                    # topics
                    real_topics += self.get_multi_label_class(batch_y_topics)
                    predicted_topics += self.get_multi_label_class(out_topics)

                    # emotions
                    real_emotion += self.get_multi_label_class(batch_y_emotions)
                    predicted_emotion += self.get_multi_label_class(out_emotion)

                    # speech acts
                    real_speech_acts += self.get_multi_label_class(batch_y_speech_acts)
                    predicted_speech_acts += self.get_multi_label_class(out_speech_acts)

                    seq_len.append(batch_seq_length)

                    # RESET BATCH
                    batch_x = []
                    batch_seq_length = []
                    batch_y_sentiment = []
                    batch_y_topics = []
                    batch_y_emotions = []
                    batch_y_speech_acts = []

            filtered_label_sentiment, filtered_predicted_sentiment = self.remove_extra(real_sentiment,
                                                                                       predicted_sentiment,
                                                                                       seq_len)
            filtered_label_topics, filtered_predicted_topics = self.remove_extra(real_topics, predicted_topics, seq_len)
            filtered_label_emotion, filtered_predicted_emotion = self.remove_extra(real_emotion, predicted_emotion,
                                                                                   seq_len)
            filtered_label_speech_acts, filtered_predicted_speech_acts = self.remove_extra(real_speech_acts,
                                                                                           predicted_speech_acts,
                                                                                           seq_len)
            print("====>SENTIMENT<====")
            self.measure(filtered_label_sentiment, filtered_predicted_sentiment)
            print("====>TOPICS<====")
            self.measure_multi(filtered_label_topics, filtered_predicted_topics, "topics")
            print("====>EMOTIONS<====")
            self.measure_multi(filtered_label_emotion, filtered_predicted_emotion, "emotions")
            print("====>SPEECH ACTS<====")
            self.measure_multi(filtered_label_speech_acts, filtered_predicted_speech_acts, "speech_act")
            print("\033[91m{}\033[00m".format("[" + type + " SET] Evaluation Finished!"))
            print("")

    def remove_extra(self, label, predicted, sequence):
        flat_sequence = [item for sublist in sequence for item in sublist]
        filtered_label = []
        filtered_predicted = []
        for i in range(0, len(flat_sequence)):
            start_index = i * self.params["post_padding_size"]
            end_index = start_index + self.params["post_padding_size"]
            each_label = label[start_index:end_index]
            each_predicted = predicted[start_index:end_index]
            filtered_label += each_label[:flat_sequence[i]]
            filtered_predicted += each_predicted[:flat_sequence[i]]
        # print("    Filtered results size : {}/{}".format(len(filtered_label), len(filtered_predicted)))
        return filtered_label, filtered_predicted

    def measure(self, label, predicted):
        if self.params["binary_sentiment"]:
            avg_type = "binary"
        else:
            avg_type = "macro"
        accuarcy_score = accuracy_score(label, predicted)
        print("Accuracy:  {}".format(accuarcy_score))
        print("Precision: {}".format(precision_score(label, predicted, average=avg_type)))
        print("Recall:    {}".format(recall_score(label, predicted, average=avg_type)))
        print("F-measure: {}".format(f1_score(label, predicted, average=avg_type)), )
        matrix = np.array(confusion_matrix(label, predicted))
        if self.params["binary_sentiment"]:
            print("Confusion matrix:\n{}\n{}".format(matrix[0], matrix[1]))
        else:
            print("Confusion matrix:\n{}\n{}\n{}".format(matrix[0], matrix[1], matrix[2]))

    def measure_multi(self, label, predicted, type):
        counter = 0
        acc = 0
        for i in range(0, len(label)):
            counter += 1
            count = 0
            max = np.maximum(len(label[i]), len(predicted[i]))
            for each in label[i]:
                if each in predicted[i]:
                    count += 1
            if max != 0:
                acc += count / max

        print("Accuracy:  {}".format(acc / counter))

    def get_class(self, output):
        labels = []
        for batch in output:
            for out in batch:
                labels.append(np.argmax(out))
        return labels

    def get_multi_label_class(self, output):
        labels = []
        for batch in output:
            for out in batch:
                temp = []
                for i in range(0, len(out)):
                    if out[i] >= 0.5:
                        temp.append(i)
                labels.append(temp)
        return labels
