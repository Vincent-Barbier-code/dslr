import numpy as np
import pandas as pd

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def cost_function(X, y, theta):
	m = len(y)
	h = sigmoid(np.dot(X, theta))
	cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
	return cost

class LogisticRegression:
	def __init__(self, lr=0.01, iters=1000) -> None:
		self.lr = lr
		self.iters = iters
		self.w = {}
		self.b = 0

	def fit(self, X, y):
		m, n = X.shape
		for house in np.unique(Y):
			y_all = np.where(y == house, 1, 0)
			w = np.random.rand(n)
			# w = np.zeros(X.shape[1])
			for _ in range(self.iters):
				predictions = sigmoid(np.dot(X, w))
				gradient = (1 / m) * np.dot(X.T, (predictions - y_all))
				w -= self.lr * gradient
			self.w[house] = w

	def score(self, X, y):
		return sum(self.predict(X) == y) / len(y)
	
	def score(self, X, y):
		return np.mean(self.predict(X) == y)

	def predict(self, X):
		predictions = np.zeros((X.shape[0], len(self.w)))
		for i, theta in self.w.items():
			predictions[:, i] = sigmoid(np.dot(X, theta))
		return np.argmax(predictions, axis=1)


data = pd.read_csv('../datasets/dataset_train.csv')
data = data.dropna()
data = data.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)

houses = data['Hogwarts House'].unique()


X = data.drop('Hogwarts House', axis=1)
Y = pd.factorize(data['Hogwarts House'])[0]

X = (X - X.mean()) / X.std()

m = LogisticRegression()
m.fit(X, Y)
print("Accuracy: ", m.score(X, Y))

