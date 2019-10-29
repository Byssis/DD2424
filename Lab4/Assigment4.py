import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
class RNN():
    def __init__(self, M=100, K=28, eta=0.1, seq_len=25, sigma=0.01):
        self.b = np.zeros((M, 1))
        self.c = np.zeros((K, 1))
        self.U = np.random.random((M, K)) * sigma
        self.W = np.random.random((M, M)) * sigma
        self.V = np.random.random((K, M)) * sigma
        self.grads = Gradients(M = M, K = K)
        self.adaW = np.zeros((M, M))
        self.adaU = np.zeros((M, K))
        self.adaV = np.zeros((K, M))
        self.adab = np.zeros((M, 1))
        self.adac = np.zeros((K, 1))
        self.h = np.zeros_like(self.b)
        self.eta = eta

    def evaluate(self,X, h):
        out = np.zeros_like(X)
        ht = {}
        ht[-1] = np.copy(h)
        for i in range(X.shape[1]):
            x = X[:, i].reshape((X.shape[0], 1))
            a = self.W @ ht[i-1] + self.U @ x + self.b
            ht[i] = np.tanh(a)
            o = self.V @ ht[i] + self.c
            p = soft_max(o)
            out[:, i] = p.reshape(-1)
        return out, ht

    def backwards_pass(self, X, Y, P, h):
        self.grads.reset()
        dh_next = np.zeros_like((self.b))
        for t in reversed(range(X.shape[1])):
            x = X[:, t].reshape((X.shape[0], 1))

            g = -(Y[:, t] - P[:, t]).reshape((Y.shape[0], 1))
            self.grads.V += g @ h[t].T
            self.grads.c += g

            delta_h = (1- h[t] ** 2) * (self.V.T @ g  + dh_next)
            # gradient with respect to U
            self.grads.U += delta_h @ x.T

            # gradient with respect to W
            self.grads.W += delta_h @ h[t-1].T

            # gradient with respect to b
            self.grads.b += delta_h

            # update delta_h
            dh_next = self.W.T @ delta_h

        # Clip
        self.grads.W = np.clip(self.grads.W, -5, 5)
        self.grads.V = np.clip(self.grads.V, -5, 5)
        self.grads.U = np.clip(self.grads.U, -5, 5)
        self.grads.b = np.clip(self.grads.b, -5, 5)
        self.grads.c = np.clip(self.grads.c, -5, 5)
        return h[-1]

    def adagradUpdate(self):
        e = 1e-8

        self.adaW += self.grads.W * self.grads.W
        self.W -= self.eta * self.grads.W / np.sqrt(self.adaW + e)

        self.adaV += self.grads.V * self.grads.V
        self.V -= self.eta * self.grads.V / np.sqrt(self.adaV + e)

        self.adaU += self.grads.U * self.grads.U
        self.U -= self.eta * self.grads.U / np.sqrt(self.adaU + e)

        self.adab += self.grads.b * self.grads.b
        self.b -= self.eta * self.grads.b / np.sqrt(self.adab + e)

        self.adac += self.grads.c * self.grads.c
        self.c -= self.eta * self.grads.c / np.sqrt(self.adac + e)

    def synthesize(self, start_char_index, n = 100, h=None):
        x = np.zeros_like(self.c)
        x[start_char_index] = 1
        for i in range(n):
            p, h = self.evaluate(x, h)
            #print(h.keys())
            choice = sample(p[:,0])
            yield choice
            x = np.zeros_like(self.c)
            x[choice] = 1
            h = h[0]

    def loss(self, X, Y, h_start):
        l_cross = (Y * self.evaluate(X, h_start)[0]).sum(axis=0)
        l_cross[l_cross == 0] = np.finfo(float).eps
        return np.sum(-np.log(l_cross))

    def train(self, X, Y, h_start):
        start = time.time()
        p, h_new= self.evaluate(X, h_start)
        h = self.backwards_pass(X, Y, p, h_new)
        self.h = h

        self.adagradUpdate()
        end = time.time()
        #print(end - start)

        return self.loss(X,Y, h_start), h_new

class Gradients:
    def __init__(self, M=100, K=28):
        self.K = K
        self.M = M

        self.b = np.zeros((M, 1))
        self.c = np.zeros((K, 1))

        self.U = np.zeros((M, K))
        self.W = np.zeros((M, M))
        self.V = np.zeros((K, M))

    def reset(self):
        self.b = np.zeros((self.M, 1))
        self.c = np.zeros((self.K, 1))

        self.U = np.zeros((self.M, self.K))
        self.W = np.zeros((self.M, self.M))
        self.V = np.zeros((self.K, self.M))

def sample(p):
    choice = np.random.choice(len(p), 1,p=p)
    return choice[0]
def soft_max(s):
    s_exp = np.exp(s - np.max(s))
    return s_exp / np.sum(s_exp, axis=0)
def read_file(file):
    data = []
    with open("goblet_book.txt", 'r', encoding='UTF-8') as f:
        for line in f:
            for char in line.strip():
                data.append(char)
    return data

def create_index(file):
    set_of_chars = set()
    char_to_index, index_to_char = {}, {}
    for char in read_file(file):
        set_of_chars.add(char)
    for i,c in enumerate(list(set_of_chars)):
        char_to_index[c] = i
        index_to_char[i] = c
    return char_to_index, index_to_char

def to_matrix(X, char_to_index):
    result = np.zeros((len(char_to_index.keys()), len(X)))
    for i, c in enumerate(X):
        result[ char_to_index[c], i] = 1
    return result

def train(data):
    char_to_index, index_to_char = create_index(data)
    X = data[:-1]
    Y = data[1:]
    rnn = RNN(K=len(char_to_index.keys()))
    updates = 0
    losses = []
    smooth_loss = - np.log(1.0/len(char_to_index.keys()))*25
    print(len(X))
    time_c = 0
    for i in range(3):
        h = np.zeros_like(rnn.b)
        
        for batch in tqdm(range(0, len(X), 25)):
            start = time.time()
            h_prev = h
            x = to_matrix(X[batch:batch+25], char_to_index)
            y = to_matrix(Y[batch:batch+25], char_to_index)
            loss, h = rnn.train(x,y,h)
            end = time.time()
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            losses.append(smooth_loss)
            time_c += end - start
            if updates % 10000 == 0:
                print(i,updates, end='\t')
                print(smooth_loss, end='\t')
                for char in rnn.synthesize(char_to_index[X[batch]], n = 200, h=h_prev):
                    print(index_to_char[char], end='')
                print()
                print((time_c) / 10000, 'ms')
                time_c = 0
            updates += 1
            h = h[x.shape[1] - 1]
    plt.plot(list(range(len(losses))), losses)
    plt.show()
    print("Final result")
    for char in rnn.synthesize(char_to_index[X[0]], n = 1000, h=np.zeros_like(rnn.b)):
        print(index_to_char[char], end='')

data = read_file("goblet_book.txt")            
train(data)