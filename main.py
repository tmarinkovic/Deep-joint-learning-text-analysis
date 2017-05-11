from data.Resources import Resources
from models.Model import Model

if __name__ == '__main__':
    """
    Tested on python version : 3.5.2
    """

    # parameters
    params = {
        "max_epoch": 60,
        "learning_rate": 0.001,
        "batch_size": 8,
        "post_padding_size": 10,
        "comment_padding_size": 20,
        "n_hidden": 100,
        "num_filters": 150,
        "filter_sizes": [3, 4, 5],
        "keep_prob_global_train": 0.6,
        "bidirectional": False,
        "binary_sentiment": False,
        "display_step": 100,
        "evaluate_every": 1,
        # Constants
        "word2vec_dim": 300,
        "n_classes_topics": len(Resources.topics()),
        "n_classes_emotion": len(Resources.emotions()),
        "n_classes_speech_acts": len(Resources.speech_acts()),
    }
    params["n_classes_sentiment"] = len(Resources.binary_sentiment()) if params["binary_sentiment"] is True else len(
        Resources.sentiment())

    model = Model(params)
    model.start()
