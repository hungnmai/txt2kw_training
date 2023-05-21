import re
import validators
from typing import List
from text_processing import tokenize

URL_REGEX = r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))"""


def is_url_or_domain(s: str) -> bool:
    """
    Check string is domain or url
    :param s: input
    :return: True or False
    """
    if re.match(URL_REGEX, s):
        return True
    if validators.domain(s):
        return True

    return False


def extract_domain_url(title: str) -> List:
    """
    Extract domain or url from title
    :param title:
    :return:
    """
    tokens, _ = tokenize(title)
    url_domain_list = []
    for tok in tokens:
        if is_url_or_domain(tok):
            url_domain_list.append(tok)

    return list(set(url_domain_list))


if __name__ == '__main__':
    print(is_url_or_domain("https://danmarkpotenspiller.com/kob-viagra-piller.html"))
    txt = """vnexpess.com viagra should never be used by patients taking nitrate-based medications as 
    it reported on https://danmarkpotenspiller.com/kob-viagra-piller.html. they contain nitroglycerin.
     if you take viagra in combination with drugs containing nitrates, 
     your blood pressure may drop to life-threatening levels."""

    print(extract_domain_url(txt))
