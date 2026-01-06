type("Hello, World!") #per vedere tipo di dato
float(42) #anche int e str per convertire
import math #per usare funzioni matematiche (math.sqrt(16), math.pi)
#ciclo con numeri
for i in range(3):
    print(i)
    
#f-string(formatted string literals)
d = {'one': 1}
l = [1, 2, 3]
f'dict: {d}, list: {l}, sum list: {sum(l)}'

#controlla se oggetti sono uguali
a = 'apple'
b = 'apple'
a is b #True

a,b=1,2 #assegnameno compatto
a, b = b, a #scambio valori

#ciclo su parola
word="ciao"
for letter in word:
    if letter == 'E' or letter == 'e':
        print("Trovata una E")

#condizioni compatte
def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)

#ciclo su file
for line in open("words.txt"):
    print(line)
f = open("words.txt")
f.readline()#legge una riga
f.readlines()#legge tutte le righe
f.close()#chiude il file
f.write("Ciao")#scrive nel file
print(open('deleteme.txt').read())#funzione che legge e stampa il contenuto di un file

#cicli compatti
def capitalize_title(title):
    t = [
    word.capitalize() for word in title.split()
    ]
    return ' '.join(t)

#scrittura e lettura file json
import json
writer = open('config.json', 'w')
config = {'config_1': 1, 'config_2': 2}
writer.write(json.dumps(config))
writer.close()
json.loads(open('config.json').read())

#esempio di docstring
def uses_any(word, letters):
    """
    Check if a word uses any of a list of letters
    >>> uses_any('banana', 'aeiou')
    True
    >>> uses_any('apple', 'xyz')
    False
    """
    for letter in word.lower():
        if letter in letters.lower():
            return True
    return False
if __name__ == "__main__":
    import doctest
    doctest.testmod()

#indici su stringhe
fruit = 'apple'
fruit[0:3] #app
fruit[1] #p
fruit[-1] #e
fruit[:3]#app
fruit[3:]#le

#argomenti variabili
def mean(*args):
    return sum(args) / len(args)









#lista
t = ['spam', 2.0, 5, [10, 20]]
numbers = [1, 2, 3, 4, 5]
sum(numbers) #15
max(numbers) #5
min(numbers) #1
len(numbers) #5
sorted(numbers) #[1, 2, 3, 4, 5]
numbers.append(6) #[1, 2, 3, 4, 5, 6]
numbers.insert(0, 0) #[0, 1, 2, 3, 4, 5, 6]
numbers.pop() #6 e numbers ora è [0, 1, 2, 3, 4, 5]
numbers.pop(0) #0 e numbers ora è [1, 2, 3, 4, 5]
numbers.remove(3) #[1, 2, 4, 5]
numbers.index(4) #2
numbers.extend([7, 8, 9]) #[1, 2, 4, 5, 7, 8, 9]
numbers.count(2) #1


#dizionari
numbers = {} # or numbers = dict()
numbers['zero'] = 0
numbers['one'] = 1
numbers['two'] = 2
numbers # {'zero': 0, 'one': 1, 'two': 2
numbers.keys() #dict_keys(['zero', 'one', 'two'])
numbers.values() #dict_values([0, 1, 2])
numbers.items() #dict_items([('zero', 0), ('one', 1), ('two', 2)])
len(numbers) #3
'one' in numbers #True
for k in numbers:
    print(k, numbers[k])
    
for v in numbers.values():
    print(v)

for k,v in numbers.items():
    print(k, v)
    

#defaultdict e Counter
import collections
d = collections.defaultdict(int) #default value 0
c=collections.Counter() #count elements

#tuple(immutabili come le stringhe)
t = ('spam', 2.0, 5, (10, 20))
type(t) #tuple
t[0] #spam
t2=(1,) #tupla con un solo elemento
t+t2 #(spam, 2.0, 5, (10, 20), 1)


#set (elementi unici)
s = set()
type(s)






#interazione con il sistema operativo
import os
os.getcwd() #current working directory
os.chdir('C:/') #change directory
os.listdir() #list directory contents
os.path.exists('C:/path/to/file') #check if path exists
os.path.isfile('C:/path/to/file') #check if it's a file
os.path.isdir('C:/path/to/directory') #check if it's a directory
os.path.join('C:/path', 'to', 'file') #join paths


#walk a directory
def walk(dirname):
    for name in os.listdir(dirname):
        path = os.path.join(dirname, name)
        if os.path.isfile(path):
            print(path)
        elif os.path.isdir(path):
            walk(path)











#copio classi
import copy
lunch = Time()
dinner = copy.copy(lunch)

#classi
class Time:
    """Represent a time of day"""
    def print_time(time): #metodo di istanza
        s = (
            f"{time.hour:02d}:"
            f"{time.minute:02d}:"
            f"{time.second:02d}"
        )
        print(s)
    
    #str special method
    def __str__(self):
        return (
        f"{self.hour:02d}:"
        f"{self.minute:02d}:"
        f"{self.second:02d}"
        )
    
    #costruttore
    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second

    def make_time(hour, minute, second):#funzione
        time = Time()
        time.hour = hour
        time.minute = minute
        time.second = second
        return time
    
    def int_to_time(seconds):#metodo staico(non ha self)
        minute, second = divmod(seconds, 60)
        hour, minute = divmod(minute, 60)
        return make_time(hour, minute, second)
    
    def increment_time(time, hours, minutes, seconds):
        time.hour += hours
        time.minute += minutes
        time.second += seconds
        carry, time.second = divmod(time.second, 60)
        carry, time.minute = divmod(
        time.minute + carry, 60
        )
        carry, time.hour = divmod(time.hour + carry, 60)
    
    def add_time(time, hours, minutes, seconds):
        total = copy.copy(time)
        increment_time(
        total,
        hours,
        minutes,
        seconds
        )
        return total
    
    #overload operatori (__add__, __eq__, __sub__, __lt__, __le__, __gt__, __ge__, __ne__)
    def __eq__(self, other):
        return (
        self.hour == other.hour
        and self.minute == other.minute
        and self.second == other.second
        )

start=Time()
start.print_time()
Time.print_time(start)

type(Time) #<class 'type'>
isinstance(Time(), Time) #True
hasattr(Time(), 'hour') #False
vars(Time()) #{}

#ereditarietà (hand eredita da deck)
class Card:
    """
    Represent a standard playing card
    """

    suits = ["Clubs", "Diamonds", "Hearts", "Spades"]
    ranks = [
        None,
        "Ace",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "Jack",
        "Queen",
        "King",
        "Ace",
    ]

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        rank_name = Card.ranks[self.rank]
        suit_name = Card.suits[self.suit]
        return f"{rank_name} of {suit_name}"

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def to_tuple(self):
        return (self.suit, self.rank)

    def __lt__(self, other):
        # Suit is more important than rank
        # If suits are equal, compare ranks
        return self.to_tuple() < other.to_tuple()

    def __le__(self, other):
        return self.to_tuple() <= other.to_tuple()
    
class Deck:
    """
    Represent a deck of cards
    """

    def __init__(self, cards):
        self.cards = cards

    def __str__(self):
        res = []
        for card in self.cards:
            res.append(str(card))
        return "\n".join(res)

    def make_cards():
        cards = []
        for suit in range(4):
            # Aces outrank kings
            for rank in range(2, 15):
                card = Card(suit, rank)
                cards.append(card)
        return cards

    def shuffle(self):
        random.shuffle(self.cards)

    def sort(self):
        self.cards.sort()

    def take_card(self):
        return self.cards.pop()

    def put_card(self, card):
        self.cards.append(card)

    def move_cards(self, other, num):
        for _ in range(num):
            card = self.take_card()
            other.put_card(card)


class Hand(Deck):
    """
    Represent a hand of a player
    """

    def __init__(self, label=""):
        self.label = label
        self.cards = []