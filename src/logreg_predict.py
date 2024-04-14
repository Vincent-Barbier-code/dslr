import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
import csv
import argparse
import sys
from extract import Extract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the testing dataset")
    parser.add_argument("weight_path", type=str, help="path to the weights file")
    args = parser.parse_args()

    extractor = Extract(args.path)
    extractor.dropColumns(
        ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]
    )
    extractor.fillNaN()
    x = extractor.get_data_test()

    model = LogisticRegression()
    model.load(args.weight_path)

    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    predictions = [houses[h] for h in model.predict(x)]

    with open("houses.csv", "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["Index", "Hogwarts House"])
        for i, prediction in enumerate(predictions):
            writer.writerow([i, prediction])

    print("Predictions stored in houses.csv!")


if __name__ == "__main__":
    main()
