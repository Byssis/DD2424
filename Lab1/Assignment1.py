import pickle
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

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
	n, bins, patches = plt.hist(histogram, bins='auto', color='b', alpha=0.7, rwidth=0.85)
	plt.savefig('Result_Pics/' + file_name) # save the figure to file
	plt.close() 

def EvaluateClassifier(X,W,b):
	s = W @ X.T + b 
	return soft_max(s)

def soft_max(s):
	s_exp = np.exp(s - np.max(s))
	return s_exp / np.sum(s_exp, axis=0)

def ComputeCost(X,Y,W,b,lam):
	l2 = lam * np.sum(W**2)
	l_cross = (Y.T * EvaluateClassifier(X,W,b)).sum(axis=0)
	l_cross[l_cross == 0] = np.finfo(float).eps
	return np.sum(-np.log(l_cross))/X.shape[0] + l2

def ComputeAccuracy(X,y,W,b):
	return np.sum(np.argmax(EvaluateClassifier(X,W,b), axis=0) == y) / X.shape[0]

def ComputeGradients(X,Y,P,W,b,lam):
	N = X.shape[0] 
	gradW = np.zeros_like(W)
	gradB = np.zeros_like(b)	
	g = -(Y.T-P)	
	gradW += (g @ X) / N + 2*lam*W
	gradB += g @ np.ones((N,1)) / N 
	return gradW, gradB

'''
Gradients calculations based on given source code
'''
def compute_gradients_num(X,Y,P,W,b,lam):
	h = 1e-6
	grad_W = np.zeros_like(W)
	grad_b = np.zeros_like(b)
	cost = ComputeCost(X,Y,W,b,lam)
	for i in tqdm(range(b.shape[0])):
		b[i] += h
		cost2 = ComputeCost(X,Y,W,b,lam)
		grad_b[i] = (cost2 - cost) / h
		b[i] -= h

	for i in tqdm(range(W.shape[0])):
		for j in range(W.shape[1]):
			W[i,j] += h
			cost2 = ComputeCost(X,Y,W,b,lam)
			grad_W[i, j] = (cost2 - cost) / h
			W[i,j] -= h
	return grad_W, grad_b

def test_grad():
	traning = read_images("Datasets/cifar-10-batches-py/data_batch_1")
	W = np.random.normal(0, 0.1, (10,3072))
	b = np.random.normal(0, 0.1, (10,1)) 
	lam = 10
	X = traning["input"][0:100]
	Y = traning["targets"][0:100]
	P = EvaluateClassifier(X,W,b)
	gradW, gradB = ComputeGradients(X,Y,P,W,b,lam)
	numGradW, numGradB = compute_gradients_num(X,Y,P,W,b,lam)
	print("W")
	compare(gradW, numGradW)
	print("B")
	compare(gradB, numGradB)

def compare(g, n):
	print("Avg diff {}".format(np.mean(np.abs(g - n))))
	print("Length {} {} {}".format(np.linalg.norm(g), np.linalg.norm(n), np.linalg.norm(g) - np.linalg.norm(n)))
	print("Max {} {}".format(np.max(g), np.max(n)))
	print("Min {} {}".format(np.min(g), np.min(n)))

def fit(traning, validation, batch_size=100, eta=0.1, n_epocs=20, lam=0.5, shuffle=False):
	N = traning["input"].shape[0]
	W = np.random.normal(0, 0.01, (10,3072))
	b = np.random.normal(0, 0.01, (10,1)) 
	history = np.zeros((n_epocs,4))
	for epoc in tqdm(range(n_epocs)):
		if shuffle:
			indicies = list(range(N))
			np.random.shuffle(indicies)
			traning["input"] = traning["input"][indicies]
			traning["targets"] = traning["targets"][indicies]
			traning["labels"] = traning["labels"][indicies]
		for i in range(int(N/batch_size)):
			start = batch_size*i
			end = batch_size*(i+1)
			X = traning["input"][start:end]
			Y = traning["targets"][start:end]
			P = EvaluateClassifier(X, W, b)
			gradW, gradB = ComputeGradients(X, Y, P, W, b, lam)
			W -= eta * gradW
			b -= eta * gradB
		history[epoc][0] = ComputeCost(traning["input"],traning["targets"],W,b,lam)
		history[epoc][1] = ComputeCost(validation["input"],validation["targets"],W,b,lam)
		history[epoc][2] = ComputeAccuracy(traning["input"],traning["labels"],W,b)
		history[epoc][3] = ComputeAccuracy(validation["input"],validation["labels"],W,b)
		#if epoc % 5 == 0:
		#	eta /= 10 
	return W, b, history

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
	
def experiment(name="",batch_size=100, eta=0.01, n_epocs=20, lam=0, shuffle=False, plot_graphs=False):
	traning = read_images("Datasets/cifar-10-batches-py/data_batch_1")
	validation = read_images("Datasets/cifar-10-batches-py/data_batch_2")
	test = read_images("Datasets/cifar-10-batches-py/test_batch")
	lables = get_lables("Datasets/cifar-10-batches-py/batches.meta")
	W, b, history = fit(traning, validation, batch_size=batch_size, eta=eta, n_epocs=n_epocs, lam=lam, shuffle=shuffle)
	if plot_graphs:
		plot([history[:,0], history[:,1]], "Cost" + name + ".png", ylabel="Cost")
		plot([history[:,2], history[:,3]], "Accuracy" + name + ".png", ylabel="Accuracy")
		plot_weights(W, lables,"Weights" + name + ".png")
		plot_histogram(W, "WeightsHistogram" + name + ".png")
	print("{} batch_size={}, eta={}, n_epocs={}, lam={}, shuffle={}".format(name, batch_size, eta, n_epocs,lam, shuffle))
	print(ComputeAccuracy(traning["input"],traning["labels"],W,b) , ComputeCost(traning["input"],traning["targets"],W,b, lam))
	print(ComputeAccuracy(validation["input"],validation["labels"],W,b),ComputeCost(validation["input"],validation["targets"],W,b, lam))
	print(ComputeAccuracy(test["input"],test["labels"],W,b), ComputeCost(test["input"],test["targets"],W,b, lam)) 

def experiment_alldata(name="",batch_size=100, eta=0.01, n_epocs=20, lam=0, shuffle=False, plot_graphs=False):
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
	lables = get_lables("Datasets/cifar-10-batches-py/batches.meta")
	W, b, history = fit(traning, validation, batch_size=batch_size, eta=eta, n_epocs=n_epocs, lam=lam, shuffle=shuffle)
	if plot_graphs:
		plot([history[:,0], history[:,1]], "Cost" + name + ".png", ylabel="Cost")
		plot([history[:,2], history[:,3]], "Accuracy" + name + ".png", ylabel="Accuracy")
		plot_weights(W, lables,"Weights" + name + ".png")
		plot_histogram(W, "WeightsHistogram" + name + ".png")
	print("{} batch_size={}, eta={}, n_epocs={}, lam={}, shuffle={}".format(name, batch_size, eta, n_epocs,lam, shuffle))
	print(ComputeAccuracy(traning["input"],traning["labels"],W,b) , ComputeCost(traning["input"],traning["targets"],W,b, lam))
	print(ComputeAccuracy(validation["input"],validation["labels"],W,b),ComputeCost(validation["input"],validation["targets"],W,b, lam))
	print(ComputeAccuracy(test["input"],test["labels"],W,b), ComputeCost(test["input"],test["targets"],W,b, lam)) 

#test_grad()
experiment(name="Experiment1",batch_size=100, eta=0.1, n_epocs=40, plot_graphs=True)
experiment(name="Experiment2",batch_size=100, eta=0.01, n_epocs=40, plot_graphs=True)
experiment(name="Experiment3",batch_size=100, eta=0.01, n_epocs=40, lam=0.1, plot_graphs=True)
experiment(name="Experiment4",batch_size=100, eta=0.01, n_epocs=40, lam=1, plot_graphs=True)	