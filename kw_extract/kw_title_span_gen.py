import random
from text_processing import tokenize


def is_punct(word):
    if word in ["'", ".", "-", "!", '"', '?']:
        return True
    return False


def gen_1_span(p_indices, leng, words):
    result = []
    for start, end in p_indices:
        inter = end - start + 1
        if inter <= leng - 5:
            n_start, n_end = start, end
            if start > 3:
                cand_inds = list(range(start - 3, start + 1))
                cand_inds = [ind for ind in cand_inds if not is_punct(words[ind])]
                n_start = random.choice(cand_inds)
            if end < leng - 3:
                cand_inds = list(range(end, end + 4))
                cand_inds = [ind for ind in cand_inds if not is_punct(words[ind])]
                n_end = random.choice(cand_inds)
            result.append([(n_start, n_end)])
    random.shuffle(result)
    if len(result) > 6:
        return result[: 6]
    return result


def gen_2_span(p_indices):
    if len(p_indices) == 1:
        return []
    result = []
    size = len(p_indices)
    pairs = []
    for i in range(size - 1):
        for j in range(i + 1, min(size, i + 3)):
            s1, e1 = p_indices[i]
            s2, e2 = p_indices[j]
            if s2 >= e1 + 3:
                pairs.append((i, j))
    for i, j in pairs:
        m = random.randint(0, i)
        n = random.randint(j, size - 1)
        span1 = (p_indices[m][0], p_indices[i][1])
        span2 = (p_indices[j][0], p_indices[n][1])
        result.append((span1, span2))
    if len(result) > 6:
        random.shuffle(result)
        return result[:6]
    if len(result) == 0:
        result.append((p_indices[0], p_indices[1]))
    return result


def gen_3_span(p_indices):
    if len(p_indices) < 3:
        return []
    size = len(p_indices)
    pairs = []
    for i in range(size - 2):
        for j in range(i + 1, size - 1):
            for k in range(j + 1, size):
                s1, e1 = p_indices[i]
                s2, e2 = p_indices[j]
                s3, e3 = p_indices[k]
                if s2 - e1 >= 3 and s3 - e2 >= 3:
                    pairs.append((i, j, k))
    result = []
    for i, j, k in pairs:
        m = random.randint(0, i)
        n = random.randint(k, size - 1)
        span1 = (p_indices[m][0], p_indices[i][1])
        span3 = (p_indices[k][0], p_indices[n][1])
        span2 = p_indices[j]
        result.append((span1, span2, span3))
    random.shuffle(result)
    if len(result) > 6:
        return result[6:]
    if len(result) == 0:
        result.append((p_indices[0], p_indices[1], p_indices[2]))
    return result


def connect_phrases(indices):
    result = []
    i = 0
    size = len(indices)
    while i < size:
        start = indices[i]
        k = 0
        while i + k < size and start + k == indices[i + k]:
            k += 1
        end = start + k - 1
        i = i + k
        result.append((start, end))
    return result


def gen_training_example(kw, title, verbose=False):
    words, indices = tokenize(title)
    kw_set = set(kw)
    k_indices = []
    for i in range(len(words)):
        if words[i] in kw_set:
            k_indices.append(i)
    if verbose:
        print('k_indices: ', k_indices)
    p_indices = connect_phrases(k_indices)
    if verbose:
        print('p_indices: ', p_indices)
    result = []
    l1 = gen_1_span(p_indices, len(words), words)
    result.extend(l1)
    if verbose:
        print('after gen 1: ', l1)
    l2 = gen_2_span(p_indices)
    if verbose:
        print('after gen 2: ', l2)
    result.extend(l2)
    l3 = gen_3_span(p_indices)
    result.extend(l3)
    if verbose:
        print('after gen 3: ', l3)
    f_result = []
    for spans in result:
        res = []
        for span in spans:
            start, end = span
            surface = title[indices[start][0]: indices[end][1] + 1]
            res.append(surface)
        f_result.append(res)
    return f_result


def test_generating_span():
    kw = ['viagra', 'medications', 'drugs']
    txt = """viagra should never be used by patients taking nitrate-based medications as it reported on https://danmarkpotenspiller.com/kob-viagra-piller.html they contain nitroglycerin. if you take viagra in combination with drugs containing nitrates, your blood pressure may drop to life-threatening levels."""

    list_of_spans = gen_training_example(kw, txt, verbose=True)
    for spans in list_of_spans:
        print(spans)


if __name__ == "__main__":
    test_generating_span()
