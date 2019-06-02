import numpy as np

class RNN():
    def __init__(self, M=100, K=28, eta=0.1, seq_len=25, sigma=0.1):
        self.b = np.zeros((M, 1))
        self.c = np.zeros((K, 1))
        self.U = np.random.random((M, K)) * sigma
        self.W = np.random.random((M, M)) * sigma
        self.V = np.random.random((K, M)) * sigma

    def evaluate(self,X, ht = np.zeros((100,1))):
        out = np.zeros_like(X)
        ht = np.zeros_like(self.b)
        for i in range(X.shape[1]):
            x = X[:, i:i+1]
            a = self.W @ ht + self.U @ x + self.b
            ht = np.zeros_like(a)
            np.tanh(a, out=ht)
            o = self.V @ ht + self.c
            p = soft_max(o)
            out[:, i:i+1] = p
        return out, ht

    def backwards_pass(self, X, Y, P):
        dLdO = []
        dLdh = []
        dLdO.append(-(Y[:, -1] - P[:, -1]).T)
        dLdh.append(dLdO @ self.V) 
        for t in reversed(range(X.shape[1] - 1)):
            dLdO.append(-(Y[:, t] - P[:, t]).T)
            dLdh.append(dLdO @ self.V + ) 
        pass

    def synthesize(self, start_char_index, n = 100):
        h = np.zeros_like(self.b)
        x = np.zeros_like(self.c)
        x[start_char_index] = 1
        for i in range(n):
            p, h = self.evaluate(x, ht = h)
            choice = sample(p[:,0])
            yield choice
            x = np.zeros_like(self.c)
            x[choice] = 1

    def loss(self, X, Y):
        l_cross = (Y * self.evaluate(X)[0]).sum(axis=0)
        l_cross[l_cross == 0] = np.finfo(float).eps
        return np.sum(-np.log(l_cross))


def sample(p):
    choice = np.random.choice(len(p), 1,p=p)
    return choice[0]
def soft_max(s):
    s_exp = np.exp(s - np.max(s))
    return s_exp / np.sum(s_exp, axis=0)
def read_file(file):
    with open("goblet_book.txt", 'r', encoding='UTF-8') as f:
        for line in f:
            for char in line.strip():
                yield char

def create_index(file):
    set_of_chars = set()
    char_to_index, index_to_char = {}, {}
    for char in read_file(file):
        set_of_chars.add(char)
    for i,c in enumerate(list(set_of_chars)):
        char_to_index[c] = i
        index_to_char[i] = c
    return char_to_index, index_to_char

def sequence(file, seq_len=25):
    X, Y = [], []
    i = 0
    next = None
    for char in read_file(file):
        prev = next
        next = char
        if prev != None:
            X.append(prev)
            Y.append(next)
        if len(X) == seq_len:
            assert len(X) == len(Y)
            yield X, Y
            X, Y = [], []

def to_matrix(X, char_to_index):
    result = np.zeros((len(char_to_index.keys()), len(X)))
    for i, c in enumerate(X):
        result[ char_to_index[c], i] = 1
    return result

char_to_index, index_to_char = create_index("goblet_book.txt")
#print(list(char_to_index.keys()))

rnn = RNN(K=len(char_to_index.keys()))
#for char in rnn.synthesize(char_to_index['.'], n=1000):
    #print(index_to_char[char], end='')
for seq in sequence("goblet_book.txt", seq_len=99):
   X = to_matrix(seq[0], char_to_index)
   Y = to_matrix(seq[1], char_to_index)
   p = rnn.loss(X, Y)
   print(p)
