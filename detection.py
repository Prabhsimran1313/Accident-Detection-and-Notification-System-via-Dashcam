import numpy as np
from tensorflow import keras
from keras.applications.inception_v3 import preprocess_input
import pandas as pd

IMG_SIZE = 224
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 2048

# Load your training data CSV file
train_df = pd.read_csv("train.csv")

class AccidentDetectionModel(object):
    def __init__(self, model_json_file, model_weights_file):
        # Load the model architecture
        with open(model_json_file, 'r') as json_file:
            model_json = json_file.read()
        self.loaded_model = keras.models.model_from_json(model_json)

        # Load the model weights
        self.loaded_model.load_weights(model_weights_file)

        # Load the label processor
        self.label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))

        # Load the feature extractor
        self.feature_extractor = self.build_feature_extractor()

    def predict_accident(self, img):
        img = preprocess_input(img)
        frame_features, frame_mask = self.prepare_single_video(img)
        probabilities = self.loaded_model.predict([frame_features, frame_mask])[0]
        class_vocab = self.label_processor.get_vocabulary()

        pred_class = class_vocab[np.argmax(probabilities)]
        pred_prob = np.max(probabilities)

        return pred_class, pred_prob

    def build_feature_extractor(self):
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        return feature_extractor

    def prepare_single_video(self, frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                frame_features[i, j, :] = self.feature_extractor.predict(batch[None, j, :])
            frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask