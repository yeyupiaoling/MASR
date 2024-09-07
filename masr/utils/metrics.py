from Levenshtein import distance

from masr.utils.utils import is_english_word


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.

    :param reference: The reference sentence.
    :type reference: str
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: str
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: float
    """
    # 是否要忽略大小写
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    # 是否要忽略空格
    join_char = ' '
    if remove_space:
        join_char = ''

    reference = join_char.join(list(filter(None, reference.split(' '))))
    hypothesis = join_char.join(list(filter(None, hypothesis.split(' '))))

    # 计算词错率
    edit_distance = distance(reference, hypothesis)

    if len(reference) == 0:
        raise ValueError("输入的reference为空")

    w = float(edit_distance) / len(reference)
    return w


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.

    :param reference: The reference sentence.
    :type reference: str
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: str
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: float
    """
    # 是否要忽略大小写
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    # 把所有单词都映射成字符
    words = set(reference.split(delimiter) + hypothesis.split(delimiter))
    word_dict = {w: chr(ord("一") + i) for i, w in enumerate(words)}

    # 把单词映射成字符
    ref_words_str = ''.join([word_dict[w] for w in reference.split(delimiter)])
    hyp_words_str = ''.join([word_dict[w] for w in hypothesis.split(delimiter)])

    # 计算词错率
    edit_distance = distance(ref_words_str, hyp_words_str)

    if len(hyp_words_str) == 0:
        raise ValueError("输入的reference为空")

    w = float(edit_distance) / len(hyp_words_str)
    return w


def mer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.

    :param reference: The reference sentence.
    :type reference: str
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: str
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and length of reference sentence.
    :rtype: float
    """
    # 是否要忽略大小写
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    # 把所有单词都映射成字符
    words = set()
    for w in reference.split(delimiter) + hypothesis.split(delimiter):
        if is_english_word(w):
            words.add(w)
    word_dict = {w: chr(ord("一") + i) for i, w in enumerate(words)}

    reference_new = "".join([word_dict[w] if is_english_word(w) else w for w in reference.split(delimiter)])
    hypothesis_new = "".join([word_dict[w] if is_english_word(w) else w for w in hypothesis.split(delimiter)])

    # 计算词错率
    edit_distance = distance(reference_new, hypothesis_new)

    if len(reference_new) == 0:
        raise ValueError("输入的reference为空")

    w = float(edit_distance) / len(reference_new)
    return w
