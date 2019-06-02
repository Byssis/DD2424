import pickle
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, units=10, activation="relu", input=100, lam=0, learning_rate=0.01, batch_norm=False, alpha=0.9):
        self.units = units
        self.activation = activation
        self.input = input
        self.W = np.random.normal(0, 1/np.sqrt(input), (units,input))
        self.b = np.zeros((units, 1))
        self.learning_rate = learning_rate
        self.H = None
        self.S = None
        self.Shat = None
        self.X = None
        self.Gamma = np.ones((units, 1))
        self.Beta =  np.zeros((units, 1))
        self.gradW = None
        self.gradB= None
        self.gradGamma = 0
        self.gradBeta= 0
        self.lam = lam
        self.batch_norm = batch_norm
        self.mu = []
        self.muGlobal = []
        self.var = False
        self.alpha = alpha

    def EvaluateLayer(self,X, traning=False):
        self.X = X
        self.S = np.zeros(( X.shape[0],self.units))
        self.S += (self.W @ X.T + self.b).T 
        sPrime = np.zeros_like(self.S)
        if self.batch_norm:
            self.mu = np.mean(self.S, axis=0)
            self.var = np.var(self.S, axis=0)
            self.Shat = BatchNormalize(self.S, self.mu, self.var)
            sPrime += self.Gamma.T * self.Shat + self.Beta.T 
        else:
            sPrime +=  self.S 

        if self.activation == "soft_max":
            self.H = soft_max(self.S.T).T
        else:
            self.H = relu(sPrime.T).T
        return self.H

    def ComputeGradients(self,g):
        N = self.X.shape[0] 
        self.gradW = np.zeros_like(self.W)
        self.gradB = np.zeros_like(self.b)    
       
        if self.activation == "relu":
            ind = np.where(self.S>0,1,0)[:,0]  
            g = np.diag(ind) 

        self.gradW += (g @ self.X) / N + 2*self.lam*self.W
        self.gradB += g @ np.ones((N,1)) / N 
        return self.gradW, self.gradB

    def BatchNormBackPass(self, g):
        N = g.shape[1]
        self.gradGamma = np.mean(g.T * self.Shat)
        self.gradBeta = np.mean(g.T) 
        g *= self.Gamma
        gBatch = np.zeros_like(g)
        sigma1 = ((self.var + np.finfo(float).eps)**-0.5).reshape((self.units, 1))
        sigma2 = ((self.var + np.finfo(float).eps)**-1.5).reshape((self.units, 1))
        G1 = g * sigma1
        G2 = g * sigma2
        D = (self.S - self.mu).T
        c = G2 * D
        gBatch += G1 
        gBatch -= (G1 @ np.ones((N, 1))) @  np.ones((1, N)) / N
        gBatch -= (D * (c @ np.ones((N,1)))) / N
        return gBatch

    def Update(self, learning_rate = None):
        if learning_rate != None:
            self.UpdateParms(learning_rate)
        else:
            self.UpdateParms(self.learning_rate)
    
    def UpdateParms(self, learning_rate):
        self.W -= learning_rate * self.gradW
        self.b -= learning_rate * self.gradB
        if self.batch_norm:
            self.Gamma -= learning_rate * self.gradGamma
            self.Beta -= learning_rate * self.gradBeta

def BatchNormalize(S, mu, var):
    return ((var + np.finfo(float).eps)**-0.5)*(S-mu)

def read_images(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    X = dict[b'data']
    y = dict[b'labels']
    Y = np.eye(10)[y]
    X.reshape((10000,3072))
    return { "input":X/255, "targets":Y, "labels": np.array(y)}

def read_all():
    path = "Datasets/cifar-10-batches-py/data_batch_"
    N = 5
    batches = []
    data = read_images("Datasets/cifar-10-batches-py/data_batch_1")
    for i in range(2, N + 1):
        with open(path + str(i), 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            X = dict[b'data']
            y = dict[b'labels']
            Y = np.eye(10)[y]
            X.reshape((10000,3072))
            data["input"] = np.concatenate((data["input"], X/255), axis=0)
            data["targets"] = np.concatenate((data["targets"], Y), axis=0)
            data["labels"] = np.concatenate((data["labels"], np.array(y)), axis=0)
    indicies = list(range(data["input"].shape[0]))
    np.random.shuffle(indicies)
    data["input"] = data["input"][indicies]
    data["targets"] = data["targets"][indicies]
    data["labels"] = data["labels"][indicies]
    return data

def get_normalize_paramters(data): 
    meanX = np.mean(data)
    stdX = np.std(data)
    return meanX, stdX

def normalize(data, meanX, stdX): 
    data -= meanX
    data /= stdX
    return data

def display_image(image):
    red = np.reshape(image[0:1024],(32,32))
    green = np.reshape(image[1024:2048],(32,32))
    blue =  np.reshape(image[2048:3072],(32,32))
    plt.imshow(np.dstack((red, green, blue)))
    plt.show()

def plot_weights(W, labels, file_name):
    for i, row in enumerate(W):
        img = (row - row.min()) / (row.max() - row.min())
        plt.subplot(2, 5, i+1)
        resize_image = np.reshape(img, (32, 32, 3), order='F')
        rot_imag = np.rot90(resize_image, k=3)
        plt.imshow(rot_imag, interpolation="gaussian")
        plt.axis('off')
        plt.title(labels[i].decode("utf-8"))
    plt.savefig('Result_Pics/' + file_name) # save the figure to file
    plt.close() 

def plot_histogram(W, file_name):
    histogram = W.flatten()
    plt.hist(histogram, bins='auto', color='b', alpha=0.7, rwidth=0.85)
    plt.savefig('Result_Pics/' + file_name) # save the figure to file
    plt.close() 

def EvaluateClassifier(X,layers, traning=False):
    for layer in layers:
        X = layer.EvaluateLayer(X, traning=traning)
    return X

def summary(layers):
    sum = 0
    for layer in layers:
        sum += layer.units * layer.input + layer.units
    return sum

def soft_max(s):
    s_exp = np.exp(s - np.max(s))
    return s_exp / np.sum(s_exp, axis=0)

def relu(s):
    s[s < 0] = 0
    return s

def ComputeCost(X,Y, layers):
    l2 = 0
    for layer in layers:
        l2 += layer.lam * np.sum(layer.W**2)
    l_cross = (Y * EvaluateClassifier(X,layers)).sum(axis=1)
    l_cross[l_cross == 0] = np.finfo(float).eps
    return np.sum(-np.log(l_cross))/X.shape[0] + l2

def ComputeAccuracy(X,y,layers):
    return np.sum(np.argmax(EvaluateClassifier(X,layers), axis=1) == y) / X.shape[0]


def ComputeGradientsV2(X,Y,layers):
    N = X.shape[0]
    layers[0].gradW = np.zeros_like(layers[0].W)
    layers[0].gradB = np.zeros_like(layers[0].b) 

    P = layers[-1].H
    g = (P - Y).T
    for i in reversed(range(1,len(layers))):
        layers[i].gradW = np.zeros_like(layers[i].W)
        layers[i].gradB = np.zeros_like(layers[i].b) 
        layers[i].gradGamma = 0
        layers[i].gradBeta = 0
        if layers[i].batch_norm == True:
            g = layers[i].BatchNormBackPass(g)
      
        layers[i].gradW += (g @ layers[i].X) / N 
        layers[i].gradW += 2 * layers[i].lam * layers[i].W 
        layers[i].gradB += g @ np.ones((N,1)) / N  
        g = g.T @ layers[i].W 
        ind = np.where(layers[i].X > 0, 1, 0)
        g = (g.T * ind.T)
   
    if layers[0].batch_norm == True:
        g = layers[0].BatchNormBackPass(g)
      
    layers[0].gradW += (g @ X) / N 
    layers[0].gradW += 2 * layers[0].lam * layers[0].W 
    layers[0].gradB += g @ np.ones((N,1)) / N  
    return layers

def Update(layers, eta):
    for layer in layers:
        layer.Update(eta)

def cyclic_learning_rate(eta_min, eta_max, n_s, t):
    l = t % (n_s * 2)
    if l < n_s:
        return eta_min + (l / n_s) * (eta_max -eta_min)
    else:
        return eta_max - ((l-n_s) / n_s) * (eta_max -eta_min)
'''
Gradients calculations based on given source code
'''
def compute_gradients_num(X,Y,layers):
    h = 1e-6
    for layer in layers:
        grad_W = np.zeros_like(layer.W)
        grad_b = np.zeros_like(layer.b)
        cost = ComputeCost(X,Y,layers)
        for i in tqdm(range(layer.W.shape[0])):
            for j in range(layer.W.shape[1]):
                layer.W[i,j] += h
                cost2 = ComputeCost(X,Y,layers)
                grad_W[i, j] = (cost2 - cost) / h
                layer.W[i,j] -= h
        for i in tqdm(range(layer.b.shape[0])):
            layer.b[i] += h
            cost2 = ComputeCost(X,Y, layers)
            grad_b[i] = (cost2 - cost) / h
            layer.b[i] -= h
        layer.gradB = grad_b
        layer.gradW = grad_W
    return layers[0].gradW, layers[0].gradB, layers[1].gradW, layers[1].gradB

def test_grad():
    traning = read_images("Datasets/cifar-10-batches-py/data_batch_1")
    layer1 = Layer(input=3072, units=11, lam=0)
    layer2 = Layer(input=11, units=10, lam=0)
    layers = [layer1, layer2]
    X = traning["input"][0:2]
    Y = traning["targets"][0:2]
    P = EvaluateClassifier(X,layers)
    gW1, gB1, gW2, gB2 = ComputeGradients(X,Y,layers)
    gW1num, gB1num, gW2num, gB2num  = compute_gradients_num(X,Y,layers)
    print("W1")
    compare(gW1, gW1num)
    print("B1")
    compare(gB1, gB1num)
    print("W2")
    compare(gW2, gW2num)
    print("B2")
    compare(gB2, gB2num)

def compare(g, n):
    print("Avg diff {}".format(np.mean(np.abs(g - n))))
    print("Length {} {} {}".format(np.linalg.norm(g), np.linalg.norm(n), np.linalg.norm(g) - np.linalg.norm(n)))
    print("Max {} {}".format(np.max(g), np.max(n)))
    print("Min {} {}".format(np.min(g), np.min(n)))

def fit(layers, traning, validation, batch_size=100, eta_min=1e-5, eta_max=1e-1, n_s=500, n_epocs=20, lam=0.5, shuffle=False):
    N = traning["input"].shape[0]
    history = np.zeros((n_epocs,4))
    t = 0
    for epoc in tqdm(range(n_epocs)):
        if shuffle:
            indicies = list(range(N))
            np.random.shuffle(indicies)
            traning["input"] = traning["input"][indicies]
            traning["targets"] = traning["targets"][indicies]
            traning["labels"] = traning["labels"][indicies]
        for i in range(N//batch_size):
            eta = cyclic_learning_rate(eta_min, eta_max, n_s,t) 
            start = batch_size*i
            end = batch_size*(i+1)
            X = traning["input"][start:end]
            Y = traning["targets"][start:end]
            EvaluateClassifier(X, layers, traning=True)
            ComputeGradientsV2(X, Y, layers)
            Update(layers, eta)
            t += 1
        history[epoc][0] = ComputeCost(traning["input"],traning["targets"],layers)
        history[epoc][1] = ComputeCost(validation["input"],validation["targets"],layers)
        history[epoc][2] = ComputeAccuracy(traning["input"],traning["labels"],layers)
        history[epoc][3] = ComputeAccuracy(validation["input"],validation["labels"],layers) 
    return history

def plot(series, file_name, ylabel=""):
    l = ["Train", "Validation"]
    for i,serie in enumerate(series):
        x = list(range(1,serie.shape[0] +1))
        plt.plot(x, serie, label=l[i])
        plt.legend(loc='upper right')
    plt.xlabel("epoc")
    plt.ylabel(ylabel)
    plt.savefig('Result_Pics/' + file_name) # save the figure to file
    plt.close() 

def get_lables(file):
    with open(file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return meta[b'label_names']
    
def experiment_alldata(name="",batch_size=100, eta=0.01, n_epocs=20, lam=0, shuffle=False, plot_graphs=True, eta_min=1e-5, eta_max=1e-1, n_s=None, batch_norm=False):
    data = read_all()
    traning = {}
    validation = {}
    traning["input"] = data["input"][1000:]
    traning["targets"] = data["targets"][1000:]
    traning["labels"] = data["labels"][1000:]
    validation["input"] = data["input"][:1000]
    validation["targets"] = data["targets"][:1000]
    validation["labels"] = data["labels"][:1000]    
    test = read_images("Datasets/cifar-10-batches-py/test_batch")

    # Normalize the input
    mean, std = get_normalize_paramters(traning["input"])
    traning["input"] = normalize(traning["input"], mean, std)
    validation["input"] = normalize(validation["input"], mean, std)
    test["input"] = normalize(test["input"], mean, std)

    if n_s == None:
        n_s = 2 * len(traning["input"])// batch_size
    print(n_s)
    # Define model and fit it
    layers = [
        Layer(input=3072, units=50, activation="relu", lam=lam, batch_norm=batch_norm),
        Layer(input=50, units=50, activation="relu", lam=lam, batch_norm=batch_norm),
        Layer(input=50, units=10, activation="soft_max", lam=lam),
    ]
  
    print("# parameters: ",  summary(layers))
    history = fit(layers, traning, validation, batch_size=batch_size, n_epocs=n_epocs, lam=lam, shuffle=shuffle, eta_min=eta_min, eta_max=eta_max, n_s=n_s)

    # Plot result 
    if plot_graphs:
        plot([history[:,0], history[:,1]], "Cost" + name + ".png", ylabel="Cost")
        plot([history[:,2], history[:,3]], "Accuracy" + name + ".png", ylabel="Accuracy")

    # Print result
    print("{} batch_size={}, eta={}, n_epocs={}, lam={}, shuffle={}".format(name, batch_size, eta, n_epocs, lam, shuffle))
    print(ComputeAccuracy(traning["input"],traning["labels"],layers) , ComputeCost(traning["input"],traning["targets"],layers))
    print(ComputeAccuracy(validation["input"],validation["labels"],layers),ComputeCost(validation["input"],validation["targets"],layers))
    print(ComputeAccuracy(test["input"],test["labels"],layers), ComputeCost(test["input"],test["targets"],layers)) 

def test_eta():
    n_s = 500
    eta = []
    for t in range(2*n_s):
        eta.append(cyclic_learning_rate(1e-5, 1e-1, n_s, t))
    plt.plot(list(range(2*n_s)), eta)
    plt.show()

def parameter_search(traning, validation, layers, batch_size=100, lmin=-5, lmax=-1, cycles=2, search_rounds=5):
    best = 0
    best_lam = 0
    for s in range(search_rounds):
        print("Round ", s + 1, lmin, lmax)
        candidates = [lmin + (lmax-lmin)*np.random.random() for i in range(20)]
        n_s = 2 * traning["input"].shape[0] // batch_size
        for candidate in candidates:
            network = []
            lam = 10 ** candidate
            for layer in layers:
                l = Layer(input=layer.input, units=layer.units, lam=lam, activation=layer.activation)
                network.append(l)
            _history = fit(network, traning, validation, batch_size=batch_size, n_epocs=cycles, lam=lam)
            acc = ComputeAccuracy(validation["input"],validation["labels"],network)
            print(lam, acc)
            if acc > best:
                best_lam = lam
                best = acc
        mid = np.log10(best_lam)
        diff = lmax-lmin
        lmin = mid - diff/4
        lmax = mid + diff/4

    return best_lam
    
 
experiment_alldata(name='9LayerBatch',n_epocs=20, lam=0.005, batch_size=100, eta_min=1e-5, eta_max=1e-1, shuffle=True, batch_norm=True)
#experiment(name='Figure4',n_epocs=48, lam=0.01, batch_size=100, eta_min=1e-5, eta_max=1e-1, n_s=800)
#experiment(name='Test1',n_epocs=48, lam=0.01, batch_size=100, eta_min=1e-5, eta_max=1e-1, n_s=200)