from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
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





if __name__ == "__main__":
    label, data = get_data_and_label("../data/cleaned/reviews_one")
    model = DummyModel(20000)
    model.run_modeling(label, data)

