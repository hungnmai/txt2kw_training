import re
import string


def preprocess_title(title: str, lower: bool = False) -> str:
    title = re.sub(r"\s+", " ", title)
    if lower:
        title = title.lower()

    return title


def sep_punct_from_str(token: str, remove_punt: bool = True) -> list:
    """
    Check and remove punctuation at the end of token
    :param token:
    :param remove_punt: remove punt in text
    :return:
    """
    last_idx = len(token)
    for i in range(len(token) - 1, -1, -1):
        if token[i] in string.punctuation:
            last_idx = i
        else:
            break
    if last_idx != len(token):

        return [token[0:last_idx], token[last_idx:]]
    else:
        return [token]


def tokenize(text: str):
    text = preprocess_title(title=text)
    words = text.split()
    tokens = []

    indices = []
    start = 0

    for word in words:
        items = sep_punct_from_str(word)
        tokens.extend(items)
        if len(items) == 1:
            end = start + len(items[0])
            indices.append((start, end - 1))
            start = end + 1
        else:
            end = start + len(items[0])
            indices.append((start, end - 1))
            start = end
            end = end + len(items[1])

            indices.append((start, end - 1))
            start = end + 1

    return tokens, indices


if __name__ == '__main__':
    txt = """viagra should never be used by patients taking nitrate-based medications as it reported on https://danmarkpotenspiller.com/kob-viagra-piller.html. they contain nitroglycerin. if you take viagra in combination with drugs containing nitrates, your blood pressure may drop to life-threatening levels."""

    toks, indices = tokenize(txt)
    print(toks)
    print(indices)
    i = 20
    print(indices[i], toks[i])
    print("txt: ", txt[indices[i][0]:indices[i][1] + 1])
