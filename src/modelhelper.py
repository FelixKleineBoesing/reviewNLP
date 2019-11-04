from enum import Enum
import numpy as np


class Dimension(Enum):
    d25 = 25
    d50 = 50
    d100 = 100
    d200 = 200


class GloVeEmbeddings:
    """
    word vectors published by stanford, see: https://nlp.stanford.edu/projects/glove/
    download the zip file glove.twitter.27B.zip und extract it into ~/data/
    """

    def __init__(self, path: str = "../data/", dimension: Dimension = Dimension.d100):
        self.path = path
        self.dimension = dimension
        self.file_path = "{}glove.twitter.27B.{}d.txt".format(path, dimension.value)
        self.embedding_mean = None
        self.embedding_std = None
        self.embedding_size = None
        self.number_words = None
        self.embedding_matrix = None

    def prepare_embedding_file(self, word_index, number_words: int = 20000):
        def convert_to_array(row):
            return row[0], np.array(row[1:], dtype="float32")

        with open(self.file_path, "r", encoding="utf-8") as f:
            embeddings = dict(convert_to_array(row.rstrip().rsplit(" ")) for row in f.readlines())

        embeddings_matrix = np.stack(embeddings.values())
        self.embedding_mean, self.embedding_std = embeddings_matrix.mean(), embeddings_matrix.std()
        self.embedding_size = embeddings_matrix.shape[1]
        self.number_words = min(len(word_index), number_words)
        embedding_matrix = np.random.normal(self.embedding_mean, self.embedding_std, size=(self.number_words,
                                                                                           self.embedding_size))
        for word, i in word_index.items():
            if i >= self.number_words:
                continue
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix


if __name__ == "__main__":
    embedding = GloVeEmbeddings()
    embedding.prepare_embedding_file()