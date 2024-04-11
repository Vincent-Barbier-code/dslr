import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression


def main() -> None:
    data = pd.read_csv("../datasets/dataset_train.csv")
    data = data.dropna()
    data = data.drop(
        ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Astronomy"],
        axis=1,
    )

    np.random.seed(42)
    X = data.drop("Hogwarts House", axis=1)

    facto = pd.factorize(data["Hogwarts House"])

    Y = facto[0]
    houses = facto[1].values

    print(Y, houses)

    X = (X - X.mean()) / X.std()

    # m = LogisticRegression(epochs=1)  # 475 -> 0.03
    # # try:
    # # m.load("weights.csv")
    # # except:
    # m.fit(X, Y, method="batch")
    # print("Accuracy 1: ", m.score(X, Y))

    # m2 = LogisticRegression(epochs=1)  # 480 -> 0.05
    # m2.fit(X, Y, method="mini")
    # print("Accuracy 2: ", m2.score(X, Y))

    x_train = X[:1200]
    y_train = Y[:1200]
    x_test = X[1200:]
    y_test = Y[1200:]

    m3 = LogisticRegression(epochs=1)  # 505 -> 0.98
    m3.fit(x_train, y_train, method="sto")
    print("Accuracy 3: ", m3.score(x_train, y_train))

    print("Accuracy 3: ", m3.score(x_test, y_test))


if __name__ == "__main__":
    main()
