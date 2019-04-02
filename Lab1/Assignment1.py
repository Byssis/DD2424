import pickle
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use batch 1 for traning 
# Use batch 2 for validation
# Use test for test

def read_images(file):
	with open(file, 'rb') as f:
		dict = pickle.load(f, encoding='bytes')
	X = dict[b'data']
	y = dict[b'labels']
	Y = np.eye(10)[y]
	X.reshape((10000,3072))
	return { "input":X/255, "targets":Y, "labels": y}

def display_image(image):
	red = np.reshape(image[0:1024],(32,32))
	green = np.reshape(image[1024:2048],(32,32))
	blue =  np.reshape(image[2048:3072],(32,32))

	plt.imshow(np.dstack((red, green, blue)))
	plt.show()

def display_W(W):
	plt.figure()
	for i in range(W.shape[0]):
		plt.subplot(2, W.shape[0], i + 1)
		red = np.reshape(W[i,0:1024],(32,32))
		green = np.reshape(W[i,1024:2048],(32,32))
		blue =  np.reshape(W[i,2048:3072],(32,32))
		plt.imshow(np.dstack((red, green, blue)),  interpolation='nearest')
	plt.show()

def EvaluateClassifier(X,W,b):
	s = W @ X.T + b 
	return soft_max(s)

def soft_max(s):
	s_exp = np.exp(s)
	return s_exp / np.sum(s_exp, axis=0)

def ComputeCost(X,Y,W,b,lam):
	l2 = lam * np.sum(W**2)
	l_cross = Y.T * EvaluateClassifier(X,W,b)
	l_cross[l_cross == 0] = np.finfo(float).eps
	return np.sum(-np.log(l_cross))/X.shape[0] + l2

def ComputeAccuracy(X,y,W,b):
	return np.sum(np.argmax(EvaluateClassifier(X,W,b), axis=0) == y) / X.shape[0]

def ComputeGradients(X,Y,P,W,b,lam):
	gradW = np.zeros_like(W)
	gradB = np.zeros_like(b)
	for i in range(X.shape[0]):
		x = np.reshape(X[i], (3072,1))
		y = np.reshape(Y[i], (10,1))
		p = np.reshape(P[:,i], (10,1))
		g = -(y-p)	
		gradW += (g @ x.T)
		gradB += g
	return gradW/X.shape[0] + 2*lam*W, gradB/X.shape[0]

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

def fit(traning, validation, batch_size=100, eta=0.1, n_epocs=20, lam=0.5):
	W = np.random.normal(0, 0.1, (10,3072))
	b = np.random.normal(0, 0.1, (10,1)) 
	N = traning["input"].shape[0]
	history = np.zeros((n_epocs,4))
	for epoc in tqdm(range(n_epocs)):
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
	return W, b, history

def plot(series):
	for serie in series:
		x = list(range(1,serie.shape[0] +1))
		plt.plot(x, serie)
	plt.show()

def experiment1():
	traning = read_images("Datasets/cifar-10-batches-py/data_batch_1")
	validation = read_images("Datasets/cifar-10-batches-py/data_batch_2")
	test = read_images("Datasets/cifar-10-batches-py/test_batch")
	W, b, history = fit(traning, validation, batch_size=100, eta=0.1, n_epocs=40, lam=0)
	plot([history[:,0], history[:,1]])
	plot([history[:,2], history[:,3]])
	display_W(W)
	print(ComputeAccuracy(test["input"],test["labels"],W,b)) 

def experiment2():
	traning = read_images("Datasets/cifar-10-batches-py/data_batch_1")
	validation = read_images("Datasets/cifar-10-batches-py/data_batch_2")
	test = read_images("Datasets/cifar-10-batches-py/test_batch")
	W, b, history = fit(traning, validation, batch_size=100, eta=0.01, n_epocs=40, lam=0)
	plot([history[:,0], history[:,1]])
	plot([history[:,2], history[:,3]])
	display_W(W)
	print(ComputeAccuracy(test["input"],test["labels"],W,b)) 

def experiment3():
	traning = read_images("Datasets/cifar-10-batches-py/data_batch_1")
	validation = read_images("Datasets/cifar-10-batches-py/data_batch_2")
	test = read_images("Datasets/cifar-10-batches-py/test_batch")
	W, b, history = fit(traning, validation, batch_size=100, eta=0.01, n_epocs=40, lam=0)
	plot([history[:,0], history[:,1]])
	plot([history[:,2], history[:,3]])
	display_W(W)
	print(ComputeAccuracy(test["input"],test["labels"],W,b)) 

experiment2()