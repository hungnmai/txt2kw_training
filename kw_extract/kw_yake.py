import json
import re
import yake
import uuid
import random
import os.path
import nltk
import itertools
from tqdm import tqdm
from utils import root_path
from nltk.corpus import stopwords
from typing import Any, List, Dict
from kw_extract.highligh import TextHighlighter
from text_processing import preprocess_title
from sklearn.model_selection import train_test_split
from kw_extract.url_detection import extract_domain_url

nltk.download('stopwords')
stopwords = stopwords.words('english')


def load_stopwords(path: str = os.path.join(root_path, "kw_extract/stopwords.txt")):
    with open(path, "r") as fp:
        _stopwords = [w.strip().lower() for w in fp.readlines()]
        return _stopwords


stopwords.extend(load_stopwords())


class YakeExtraction:
    """
    Using Yake algo to extract keyword
    """

    def __init__(self, max_ngram_size: int,
                 window_size: int = 1,
                 deduplication_threshold: float = 0.9,
                 deduplication_algo: str = 'seqm',
                 num_keywords: int = 3,
                 language: str = "en"):
        self.max_ngram_size = max_ngram_size
        self.kw_extractor = yake.KeywordExtractor(lan=language,
                                                  n=max_ngram_size,
                                                  dedupLim=deduplication_threshold,
                                                  dedupFunc=deduplication_algo,
                                                  windowsSize=window_size,
                                                  top=num_keywords,
                                                  features=None,
                                                  stopwords=stopwords)

    def extract_keyword(self, text: str) -> Any:
        """Extract keywords from text and return list of keywords"""
        try:
            keywords = self.kw_extractor.extract_keywords(text)
            th = TextHighlighter(max_ngram_size=self.max_ngram_size)
            txt_kw = th.highlight(text, keywords)
            pattern = r"<kw>(.*?)<\/kw>"
            matches = re.findall(pattern, txt_kw)
            return matches
        except Exception as e:
            return None


class KeywordExtraction:
    def __init__(self, max_words_in_span: int = 5):
        self.max_words_in_span = max_words_in_span

    def extract(self, text: str) -> List:
        """
        Extract keywords from title
        :param text:
        :return:
        """
        # kw_exist = ""
        spans_list = []
        for n in range(self.max_words_in_span, 0, -1):
            extractor = YakeExtraction(max_ngram_size=n)
            spans = extractor.extract_keyword(text)
            # kw_exist + " ".join(spans)
            spans_list.extend(spans)

        spans_list = list(set(spans_list))
        return spans_list


kw_extractor = KeywordExtraction()


def encode_text_to_uuid(txt: str) -> str:
    encoded_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, txt)
    return str(encoded_uuid)


def combine_keywords(keywords: list) -> Dict:
    spans = {}
    max_tokens = min(len(keywords), 3)
    for i in range(1, max_tokens + 1):
        items = [list(_item) for _item in list(itertools.combinations(keywords, i))]
        for item in items:
            item_result = []
            sorted_arr = sorted(item, key=lambda x: len(x), reverse=True)
            txt_exist = sorted_arr[0]
            item_result.append(sorted_arr[0])
            for i in range(1, len(sorted_arr)):
                if sorted_arr[i] not in txt_exist:
                    item_result.append(sorted_arr[i])
                    txt_exist += " " + sorted_arr[i]

            _id = encode_text_to_uuid(txt_exist)
            spans[_id] = item_result

    return spans


def gen_training_example(title: str):
    title = preprocess_title(title)
    url_domain_list = extract_domain_url(title)
    keywords = kw_extractor.extract(title)
    spans = combine_keywords(keywords)

    for _id, span in spans.items():
        spans[_id].extend(url_domain_list)
        random.shuffle(spans[_id])
    if len(spans) == 0:
        return None
    return {
        "spans": [values for key, values in spans.items()],
        "title": title
    }


def gen_training_data(path_read: str, path_save: str = None):
    """
    Build train and valid dataset
    :param path_read:
    :param path_save:
    :return:
    """
    items = []
    with open(path_read, "r") as fp:
        titles = [preprocess_title(title) for title in fp.readlines()]

        for title in tqdm(titles, desc="Load and build dataset"):
            item = gen_training_example(title)
            items.append(item)
    train_data, eval_data = train_test_split(items, test_size=0.1, random_state=42)

    if path_save:
        with open(os.path.join(path_save, "train_span.json", "w")) as fp:
            json.dump(train_data, fp, ensure_ascii=False, indent=4)

        with open(os.path.join(path_save, "dev_span.json", "w")) as fp:
            json.dump(eval_data, fp, ensure_ascii=False, indent=4)

    return train_data, eval_data


if __name__ == '__main__':
    txt = """viagra should never be used by patients taking nitrate-based medications as it reported on https://danmarkpotenspiller.com/kob-viagra-piller.html. they contain nitroglycerin. if you take viagra in combination with drugs containing nitrates, your blood pressure may drop to life-threatening levels."""
    txt = """how to give a woman an orgasm- step by step tips to give her a long orgasm (3 problems to overcome)"""
    spans = gen_training_example(txt)
    print(spans)

    gen_training_data("/Users/manhhung/Documents/workspace/upwork/txt2kw_training/data/title.txt",
                      "/Users/manhhung/Documents/workspace/upwork/txt2kw_training/data")
