import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import os, math, json, string, nltk
from text_processing import preprocess_title

nltk.download('stopwords')

punctuation = string.punctuation

stopwords = stopwords.words('english')

idf = pd.read_csv("idf.csv")
idf = dict(zip(idf['word'].values, idf['idf'].values))


def contains_number(token):
    return any(char.isdigit() for char in token)


def contains_punctuation(token):
    return any(p in token for p in punctuation)


def tf_idf_for_title(title: str,
                     top_k: float = 0.2,
                     min_tokens_return: int = 2,
                     max_tokens_return: int = 5,
                     verbose: bool = False):
    """
    Using tf-idf to score tokens in title
    :param title:
    :param top_k: ratio tokens return with number of tokens in title
    :param min_tokens_return: minimum tokens returned
    :param max_tokens_return: maximum tokens returned
    :param verbose: set is True to debug mode
    :return:
    """
    try:
        # preprocess title
        title = preprocess_title(title)

        tokens = title.split()
        if verbose:
            print("tokens: ", tokens)

        tokens = [tok.lower() for tok in tokens]

        tokens_freq = Counter(tokens)
        num_tokens = len(tokens_freq)
        tfidf = {}

        # if token not exist in idf
        idf_score = 6

        for tok, freq in tokens_freq.items():
            if tok not in stopwords and tok not in punctuation and contains_punctuation(tok) is False:
                if tok in idf:
                    idf_score = idf[tok]

                tfidf[tok] = (1 + math.log(1 + freq)) * idf_score

        tfidf = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
        num_tokens_return = math.ceil(top_k * num_tokens)
        if verbose:
            print("num_tokens_return: ", num_tokens_return)
        if num_tokens_return > max_tokens_return:
            num_tokens_return = max_tokens_return
        if num_tokens_return < min_tokens_return:
            num_tokens_return = min_tokens_return
        if verbose:
            print("num_tokens_return: ", num_tokens_return)

        top_keywords = tfidf[: num_tokens_return]
        if verbose:
            print("top_keywords: ", top_keywords)
        return [kw[0] for kw in top_keywords]
    except Exception as e:
        print(e)
        return []


if __name__ == "__main__":
    txt = """viagra should never be used by patients taking nitrate-based medications as it reported on danmarkpotenspiller.com. they contain nitroglycerin. if you take viagra in combination with drugs containing nitrates, your blood pressure may drop to life-threatening levels."""
    keywords = tf_idf_for_title(txt, verbose=True)
    print("keywords: ", keywords)
