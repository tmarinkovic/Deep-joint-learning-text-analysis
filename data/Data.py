import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize
from data.Resources import Resources


class Data:
    """
    Class responsible for padding, prepossessing and serving data for batching 
    """

    def __init__(self, filename, comment_padding_size, post_padding_size, word2vec_dim, binary_sentiment):
        print("Loading word2vec...")
        # word2vec model
        self.word2vec = KeyedVectors.load_word2vec_format(filename, binary=False)
        print("Done!")

        # parameters
        self.word2vec_dim = word2vec_dim
        self.comment_padding_size = comment_padding_size
        self.post_padding_size = post_padding_size
        self.binary_sentiment = binary_sentiment

        # fetch resources
        self.stop_words = Resources.stop_words()
        self.topic_one_hot = Resources.topics()
        self.emotion_one_hot = Resources.emotions()
        self.speech_acts_one_hot = Resources.speech_acts()
        self.label_one_hot = Resources.sentiment()

        # get resources length
        self.topic_count = len(Resources.topics())
        self.emotion_count = len(Resources.emotions())
        self.speech_acts_count = len(Resources.speech_acts())

        # get sentiment resource and size based on binary or not flag is set
        if binary_sentiment:
            self.labels_count = 2
            self.label_one_hot = Resources.binary_sentiment()
        else:
            self.labels_count = 3
            self.label_one_hot = Resources.sentiment()

    def preprocess(self, sentence):
        """
        Preprocess text 
        :param sentence: 
        :return purified list of word2vec vectors
        """

        # create tokens
        tokens = word_tokenize(sentence.lower())

        # clean sentences
        sentence = [word for word in tokens if word not in self.stop_words and (word.isalpha() or word.isnumeric())]

        # if not found in word2vec, remove it
        sentence = [self.word2vec[word] if word in self.word2vec.vocab else np.zeros(self.word2vec_dim) for word in
                    sentence]
        purified = [word for word in sentence if not np.array_equal(word, np.zeros(self.word2vec_dim))]

        # if needed pad comment with zeros
        sequence_length = len(purified)
        if sequence_length == 0:
            return np.zeros([self.comment_padding_size, self.word2vec_dim])
        else:
            purified = self.pad_x(purified, self.comment_padding_size, "comment")

        return purified

    def get_next(self, text):
        """
        Handles post creation and post padding
        """

        x = []
        sequence_length = 0
        y_labels_array = []
        y_topics_array = []
        y_emotions_array = []
        y_speech_act_array = []
        posts = text.rstrip("\n").split("*|*")

        for post in posts:
            if post == "":
                continue
            else:
                # fetch all features
                post_text = post.split("||")[0]
                post_topics = post.split("||")[1]
                post_emotion = post.split("||")[2]
                post_speech_acts = post.split("||")[3]
                label = post.split("||")[4]

                # preprocess it
                post_text_proccesed = self.preprocess(post_text)

                # one hot transform features
                y_topics = self.one_hot_transform(post_topics, "topic")
                y_emotion = self.one_hot_transform(post_emotion, "emotion")
                y_speech_act = self.one_hot_transform(post_speech_acts, "speech_act")
                y_sentiment = self.one_hot_transform(label, "sentiment")

                # append to placeholder
                x.append(post_text_proccesed)

                # count length for lstm
                sequence_length += 1

                # append y labels for each feature
                y_labels_array.append(y_sentiment)
                y_topics_array.append(y_topics)
                y_emotions_array.append(y_emotion)
                y_speech_act_array.append(y_speech_act)

        return self.pad_x(x, self.post_padding_size, "post"),\
               sequence_length, \
               self.pad_y(y_labels_array, self.post_padding_size, "sentiment"), \
               self.pad_y(y_topics_array, self.post_padding_size, "topic"), \
               self.pad_y(y_emotions_array, self.post_padding_size, "emotion"), \
               self.pad_y(y_speech_act_array, self.post_padding_size, "speech_act")

    '''
    def pad_sequence(self, sequence, size):
        if len(sequence) > size:
            new_text = []
            for i in range(0, size):
                new_text.append(sequence[i])
            return np.array(new_text)
        if len(sequence) < size:
            for i in range(len(sequence), size):
                sequence.append(0)
            return np.array(sequence)
        return sequence
    '''

    def pad_x(self, text, size, x_type):
        """
        Pad text to desired size
        """

        if len(text) > size:
            new_text = []
            for i in range(0, size):
                new_text.append(text[i])
            return np.array(new_text)
        if len(text) < size:
            for i in range(len(text), size):
                if x_type == "comment":
                    text.append(np.zeros(self.word2vec_dim))
                else:
                    text.append(np.zeros([self.comment_padding_size, self.word2vec_dim]))
            return np.array(text)
        return text

    def pad_y(self, data, size, y_type):
        """
        Pad labels to desired size
        """

        if y_type == "topic":
            one_hot_size = self.topic_count
        elif y_type == "emotion":
            one_hot_size = self.emotion_count
        elif y_type == "speech_act":
            one_hot_size = self.speech_acts_count
        else:
            one_hot_size = self.labels_count

        # padding
        if len(data) > size:
            new_text = []
            for i in range(0, size):
                new_text.append(data[i])
            return np.array(new_text)
        if len(data) < size:
            for i in range(len(data), size):
                data.append(np.zeros(one_hot_size))
            return np.array(data)
        return data

    def one_hot_transform(self, data, label_type):
        """
        One hot transform
        """

        if label_type == "topic":
            if data == "None":  # some weird bug
                return np.zeros(self.topic_count)
            else:
                response = np.zeros(self.topic_count)
                for each in data.split(","):
                    response[self.topic_one_hot[each.replace("|", "")]] = 1
                return response
        elif label_type == "emotion":
            if data == "None":
                return np.zeros(self.emotion_count)
            else:
                response = np.zeros(self.emotion_count)
                for each in data.split(","):
                    response[self.emotion_one_hot[each.replace("|", "")]] = 1
                return response
        elif label_type == "speech_act":
            if data == "None":
                return np.zeros(self.speech_acts_count)
            else:
                response = np.zeros(self.speech_acts_count)
                for each in data.split(","):
                    response[self.speech_acts_one_hot[each.replace("|", "")]] = 1
                return response
        else:
            response = np.zeros(self.labels_count)
            for each in data.split(","):
                response[self.label_one_hot[each.replace("|", "")]] = 1
            return response
