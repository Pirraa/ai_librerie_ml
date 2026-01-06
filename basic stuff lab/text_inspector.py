def clean_book(source_filename, destination_filename):
    """
    Remove everything except the book itself. Return the number of lines
    of the cleaned book
    source_filename : where to find the book to be cleaned
    destination_filename : where to store the cleaned book
    >>> clean_book("stevenson.txt", "stevenson_clean.txt")
    2530
    """

    start="*** START OF THIS PROJECT GUTENBERG EBOOK THE STRANGE CASE OF DR. ***"
    end="*** END OF THIS PROJECT GUTENBERG EBOOK THE STRANGE CASE OF DR. ***"
    #with open(source_filename, "r") as f:
        #lines = f.readlines()
    #with open(destination_filename, "w") as f:
        #for line in lines:
            #if line.strip("\n") != start:
                #f.write(line)

    
    #for line in f1:
        #line=line.strip("\n")
        #if line!=start:
            #continue
        #f2.write(line)
    #print(f1.readline())

    #for line in f1:
        #line=line.strip("\n")
        #if line != end:
            #f2.write(line)
        #else:
            #break
    f1=open(source_filename)
    f2=open(destination_filename,"w")

    for line in f1:
        if start in line:
            break
    for line in f1:
        if end in line:
            break
        f2.write(line)

    f1.close()
    f2.close()
    return len(open(destination_filename).readlines())


def count_unique_words(filename):
    """
    Count the number of unique words in a file
    filename : path to a file
    >>> count_unique_words("stevenson_clean.txt")
    6039
    """
    f1=open(filename)
    unique={}
    for line in f1:
        words=line.split()
        for word in words:
            unique[word]=1
    f1.close()
    return len(unique)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    #clean_book("stevenson.txt", "stevenson_clean.txt")
