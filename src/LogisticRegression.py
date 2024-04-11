import numpy as np
import csv


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

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train model with datas

        Args:
                        x (np.ndarray): Input values
                        y (np.ndarray): Target values
        """
        m, n = x.shape
        for house in np.unique(y):
            y_all = np.where(y == house, 1, 0)
            w = np.random.rand(n)
            for _ in range(self.epochs):
                predictions = sigmoid(np.dot(x, w))
                gradient = (1 / m) * np.dot(x.T, (predictions - y_all))
                w -= self.lr * gradient
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
