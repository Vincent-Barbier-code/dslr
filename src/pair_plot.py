import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import os


def main() -> None:
    if not os.path.exists('../plots/pair_plot'):
        os.makedirs('../plots/pair_plot')
    data = pd.read_csv('../datasets/dataset_train.csv')
    data = data.drop('Index', axis=1)

    sb.pairplot(data, hue="Hogwarts House")
    plt.savefig('../plots/pair_plot/answer.png')


if __name__ == "__main__":
    main()
