import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression


def main() -> None:
    data = pd.read_csv('../datasets/dataset_train.csv')
    data = data.dropna()
    data = data.drop(['Index', 'First Name', 'Last Name',
                      'Birthday', 'Best Hand'], axis=1)

    X = data.drop('Hogwarts House', axis=1)

    facto = pd.factorize(data['Hogwarts House'])

    Y = facto[0]
    houses = facto[1].values

    print(Y, houses)

    X = (X - X.mean()) / X.std()

    m = LogisticRegression()
    try:
        m.load('weights.csv')
    except:
        m.fit(X, Y)

    print("Accuracy: ", m.score(X, Y))
