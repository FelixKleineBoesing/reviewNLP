from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, Conv1D, CuDNNLSTM
from src.etl import get_data_and_label


class DummyModel:

    def __init__(self, number_words: int = 20000):
        self.number_words = number_words

    def _create_network(self):
        pass

    def run_modeling(self, label, data):
        data = self._tokenize_data(data)

    def _tokenize_data(self, data):
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(data)
        data = tokenizer.texts_to_sequences(data)
        return data

    def _train_network(self):
        pass

    def _build_network(self, batch_size: int = 1024, epochs: int = 10):
        input = Input()
        x = Embedding()(inp)
        x = Conv1D()(x)
        x = CuDNNLSTM()(x)
        x = Dense()(x)
        self.model = Model(inputs=inp, output=x)
        self.model.compile(loss="binary_crossentropy", optimizer="adam")







if __name__ == "__main__":
    label, data = get_data_and_label("../data/cleaned/reviews_one")
    model = DummyModel(20000)
    model.run_modeling(label, data)

