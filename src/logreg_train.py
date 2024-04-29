import numpy as np
from LogisticRegression import LogisticRegression
import argparse
from extract import Extract


def main() -> None:
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the training dataset")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["batch", "mini_batch", "stochastic"],
        help="training algorithm",
        default="batch",
    )
    args = parser.parse_args()

    extractor = Extract(args.path)
    extractor.dropColumns(["Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    extractor.fillNaN(["Hogwarts House"])
    x, y = extractor.get_data_training("Hogwarts House", True)

    epochs = {"batch": 1000, "mini_batch": 100, "stochastic": 1}

    model = LogisticRegression(epochs=epochs[args.mode])
    model.fit(x, y, args.mode)

    print("Weigths file created!")
    print(model.score(x, y))


if __name__ == "__main__":
    main()
