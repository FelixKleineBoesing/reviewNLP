from enum import Enum


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
        self.file_path = "{}glove.twitter.27B.{}d.zip".format(path, dimension.value)

    def prepare_embedding_file(self):
        pass