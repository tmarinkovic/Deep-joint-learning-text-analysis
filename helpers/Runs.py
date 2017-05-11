import time
import os


class Runs:
    def create_run(self):
        model_id = int(time.time())
        if not os.path.exists("runs/" + str(model_id)):
            os.makedirs("runs/" + str(model_id))
        self.model_path = "runs/" + str(model_id) + "/model.ckpt"

    def save_model(self, sess, saver):
        save_path = saver.save(sess, self.model_path)
        print("Model saved in file: %s" % save_path)
