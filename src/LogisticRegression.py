import numpy as np
import csv
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid Function

    Args:
    x (np.ndarray): Array of value

    Returns:
    np.ndarray:
    """
    return 1 / (1 + np.exp(-x))


def cost_function(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Binary Cross Entropy Loss

    Args:
    x (np.ndarray): Predicted values
    y (np.ndarray): Target values
    theta (np.ndarray): Array of weights

    Returns:
    float: Cost value
    """
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.w = {}

    def load(self, path: str) -> None:
        """Load saved weights

        Args:
        path (str): Path to weights file in csv format
        """
        with open(path, "r") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                self.w[i] = np.array([float(v) for v in row])

    def batch(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Batch gradient descent algorithm

        Args:
            x (np.ndarray): data to train with
            y (np.ndarray): labels
            w (np.ndarray): weights
        """
        predictions = sigmoid(np.dot(x, w))
        gradient = (1 / len(x)) * np.dot(x.T, (predictions - y))
        w -= self.lr * gradient

    def stochastic_gradient_descent(
        self, x: pd.DataFrame, y: np.ndarray, w: np.ndarray
    ) -> None:
        """Stochastic gradient descent algorithm

        Args:
            x (np.ndarray): data to train with
            y (np.ndarray): labels
            w (np.ndarray): weights
        """
        for i in range(len(x)):
            predictions = sigmoid(np.dot(x.values[i], w))
            gradient = np.dot(x.values[i].T, (predictions - y[i]))
            w -= self.lr * gradient

    def mini_batch(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """Mini Batch gradient descent algorithm

        Args:
            x (np.ndarray): data to train with
            y (np.ndarray): labels
            w (np.ndarray): weights
        """
        batch_size = 256
        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            predictions = sigmoid(np.dot(x_batch, w))
            gradient = (1 / len(x_batch)) * np.dot(x_batch.T, (predictions - y_batch))
            w -= self.lr * gradient

    def fit(self, x: np.ndarray, y: np.ndarray, method: str = "batch") -> None:
        """Train model with datas

        Args:
            x (np.ndarray): input values
            y (np.ndarray): target values
            method (str, optional): gradient descent algorithm used
        """
        func = None

        match method:
            case "batch":
                func = self.batch
            case "mini_batch":
                func = self.mini_batch
            case "stochastic":
                func = self.stochastic_gradient_descent
            case _:
                func = self.batch
        m, n = x.shape
        for house in np.unique(y):
            y_all = np.where(y == house, 1, 0)
            w = np.random.rand(n) - 0.5
            for _ in range(self.epochs):
                func(x, y_all, w)
            self.w[house] = w

        with open("weights.csv", "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            for h in self.w:
                writer.writerow(self.w[h])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy of predictions

        Args:
        X (np.ndarray): Predictions values
        y (np.ndarray): Target values

        Returns:
        float: Accuracy of correct value
        """
        return sum(self.predict(X) == y) / len(y)

    def predict(self, X: np.ndarray) -> int:
        """Predict label with input values

        Args:
        X (np.ndarray): Input values

        Returns:
        int: Preditected values
        """
        predictions = np.zeros((X.shape[0], len(self.w)))
        for i, theta in self.w.items():
            predictions[:, i] = sigmoid(np.dot(X, theta))
        return np.argmax(predictions, axis=1)
