

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, Conv1D, CuDNNLSTM, Input
from tensorflow.keras.models import Model

from src.etl import get_data_and_label


class DummyModel:

    def __init__(self, number_words: int = 20000, sequence_length: int = 50, embedding_size: int = 50,
                 embeddings=None):
        self.number_words = number_words
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

    def _create_network(self):
        pass

    def run_modeling(self, label, data):
        data = self._tokenize_data(data)

    def _tokenize_data(self, data):
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(data)
        data = tokenizer.texts_to_sequences(data)
        data = pad_sequences(data, maxlen=self.sequence_length)
        return data

    def _train_network(self):
        pass

    def _build_network(self, batch_size: int = 1024, epochs: int = 10):
        input = Input(shape=(self.sequence_length, ))
        x = Embedding(self.number_words, embedding_size, weights=[matr])(input)
        x = Conv1D()(x)
        x = CuDNNLSTM()(x)
        x = Dense()(x)
        self.model = Model(inputs=inp, output=x)
        self.model.compile(loss="binary_crossentropy", optimizer="adam")




if __name__ == "__main__":
    label, data = get_data_and_label("../data/cleaned/reviews_one")
    model = DummyModel(20000)
    model.run_modeling(label, data)

