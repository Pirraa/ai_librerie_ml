def check_word(word, available, required):
    """
    Check whether a word is acceptable
    word : word to check
    available : string of seven available letters
    required : string of the single required letter
    >>> check_word('color', 'ACDLORT', 'R')
    True
    >>> check_word('ratatat', 'ACDLORT', 'R')
    True
    >>> check_word('rat', 'ACDLORT', 'R')
    False
    >>> check_word('told', 'ACDLORT', 'R')
    False
    >>> check_word('bee', 'ACDLORT', 'R')
    False
    """
    if len(word)<4:
        return False
    for letter in word.lower(): 
        if letter not in available.lower():
            return False
    if required.lower() not in word.lower():
        return False
    
    return True

if __name__ == "__main__":
    import doctest
    doctest.testmod()