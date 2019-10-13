import bz2
import re
import pickle


class ETL:

    def __init__(self, file_path: str = "../data/reviews_two/train.ft.txt.bz2.bz2",
                 output_path_base: str = "../data/cleaned/train"):
        self.file_path = file_path
        self.output_path_base = output_path_base

    def run(self):
        data = self._import()
        label, data = self._clean(data)
        self._load(label, data)

    def _import(self):
        return bz2.BZ2File(self.file_path)

    def _clean(self, data):
        # convert raw to strings
        data = [row.decode("utf-8") for row in data.readlines()]
        labels = []
        sentences = []
        # split data in label and sentences
        for row in data:
            label, sentence = row.split(" ", 1)
            labels.append(0 if label == "__label__1" else 1)
            sentences.append(_clean_string(sentence[:-1]))
        return labels, sentences

    def _load(self, label, data):
        with open("{}_label.pckl".format(self.output_path_base), "wb") as f:
            pickle.dump(label, f)
        with open("{}_data.pckl".format(self.output_path_base), "wb") as f:
            pickle.dump(label, f)


def get_data_and_label(file_path_base: str):
    with open("{}_label.pckl".format(file_path_base), "w") as g:
        labels = pickle.load(g)
    with open("{}_label.pckl".format(file_path_base), "w") as g:
        data = pickle.load(g)
    return labels, data


def _clean_string(str_: str):
    str_ = str_.lower()
    # replaced every url with <url>
    if any([item in str_ for item in ["http:", "www.", "https:"]]):
        str_ = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", str_)
    return str_


if __name__ == "__main__":
    etl = ETL(file_path="../data/reviews_two/train.ft.txt.bz2.bz2",
              output_path_base="../data/cleaned/reviews_one")
    etl.run()