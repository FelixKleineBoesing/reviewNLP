import logging
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, Conv1D, LSTM, Input
from tensorflow.keras import Model

from src.etl import get_data_and_label
from src.modelhelper import GloVeEmbeddings
from src.modelhelper import Dimension


EMBEDDINGS_FILE_PATH = "../data/tokenized_data"

class DummyModel:

    def __init__(self, number_words: int = 20000, dimension: Dimension = Dimension.d100,
                 batch_size: int = 1024, epochs: int = 10, weights_file_path: str = "../data/weights/weights.hdf5",
                 caching_data: bool = False):
        self.number_words = number_words
        self.sequence_length = dimension.value
        self.embedding_size = dimension.value
        self.embeddings = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights_file_path = weights_file_path
        self.caching_data = caching_data
        self._embeddings_file_path = EMBEDDINGS_FILE_PATH

    def _create_network(self):
        pass

    def run_modeling(self, label, data):
        data = self._tokenize_data(data)
        self._build_network()
        self._train_network(data, label)

    def _tokenize_data(self, data):
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(data)
        if not self.caching_data or not os.path.isfile(self._embeddings_file_path):
            embeddings = GloVeEmbeddings()
            embeddings.prepare_embedding_file(tokenizer.word_index, self.number_words)
            self.embeddings = embeddings
            if self.caching_data:
                self.embeddings.to_file(self._embeddings_file_path)
        else:
            self.embeddings = GloVeEmbeddings().from_file(self._embeddings_file_path)

        data = tokenizer.texts_to_sequences(data)
        data = pad_sequences(data, maxlen=self.sequence_length)
        return data

    def _train_network(self, data, label):
        label = np.array(label)
        self.model.fit(data, label, batch_size=self.batch_size, epochs=self.epochs)
        self.model.save_weights(self.weights_file_path)

    def _build_network(self):
        input_ = Input(shape=(self.sequence_length, ))
        x = Embedding(self.number_words, self.embedding_size, weights=[self.embeddings.embedding_matrix])(input_)
        x = Conv1D(self.embedding_size, kernel_size=3)(x)
        x = Conv1D(128, strides=1, kernel_size=3)(x)
        x = Conv1D(128, strides=1, kernel_size=3)(x)
        x = Conv1D(256, strides=1, kernel_size=3)(x)
        x = LSTM(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=input_, outputs=x)
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
        logging.debug(self.model.summary())


if __name__ == "__main__":
    label, data = get_data_and_label("../data/cleaned/reviews_one")
    model = DummyModel(20000, caching_data=True)
    model.run_modeling(label, data)

