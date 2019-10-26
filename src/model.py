

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, Conv1D, LSTM, Input
from tensorflow.keras.models import Model

from src.etl import get_data_and_label
from src.modelhelper import GloVeEmbeddings
from src.modelhelper import Dimension

class DummyModel:

    def __init__(self, number_words: int = 20000, dimension: Dimension = Dimension.d100):
        self.number_words = number_words
        self.sequence_length = dimension.value
        self.embedding_size = dimension.value

    def _create_network(self):
        pass

    def run_modeling(self, label, data):
        data = self._tokenize_data(data)

    def _tokenize_data(self, data):
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(data)

        embeddings = GloVeEmbeddings()
        embeddings.prepare_embedding_file(tokenizer.word_index, self.number_words)

        data = tokenizer.texts_to_sequences(data)
        data = pad_sequences(data, maxlen=self.sequence_length)
        return data

    def _train_network(self):
        pass

    def _build_network(self, batch_size: int = 1024, epochs: int = 10):
        input = Input(shape=(self.sequence_length, ))
        x = Embedding(self.number_words, self.embedding_size, weights=[matrix])(input)
        x = Conv1D()(x)
        x = LSTM()(x)
        x = Dense()(x)
        self.model = Model(inputs=input, output=x)
        self.model.compile(loss="binary_crossentropy", optimizer="adam")




if __name__ == "__main__":
    label, data = get_data_and_label("../data/cleaned/reviews_one")
    model = DummyModel(20000)
    model.run_modeling(label, data)

