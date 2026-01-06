def score_word(word, available):
    """
    Compute the score for an acceptable word
    word : word to be scored
    available : string of the available letters
    >>> score_word('card', 'ACDLORT')
    1
    >>> score_word('color', 'ACDLORT')
    5
    >>> score_word('cartload', 'ACDLORT')
    15
    """
    if len(word)==4:
        return 1
    elif len(word)<7:
        return len(word)
    else:
        return len(word)+7

if __name__ == "__main__":
    import doctest
    doctest.testmod()