import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main() -> None:
    if not os.path.exists('../plots/histogram'):
        os.makedirs('../plots/histogram')
    data = pd.read_csv('../datasets/dataset_train.csv')

    hues = data["Hogwarts House"]

    data = data.drop(['Index', 'First Name', 'Last Name',
                      'Birthday', 'Best Hand', 'Hogwarts House'], axis=1)

    fig, axes = plt.subplots(5, 3)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    plt.xticks(visible=False)
    plt.subplots_adjust(bottom=0.064, top=0.945, hspace=0.5, wspace=0.4)

    for i, col in enumerate(data.columns.values[:-1]):
        sb.histplot(ax=axes[i // 3, i % 3], data=data,
                    x=col, hue=hues, legend=False, bins=50)
    last = sb.histplot(ax=axes[4, 1], data=data, x=col,
                       hue=hues, legend=True, bins=50)
    sb.move_legend(last, "upper left", bbox_to_anchor=(1.5, 1))
    axes[4, 0].set_visible(False)
    axes[4, 2].set_visible(False)
    fig.savefig('../plots/histogram/all.png')
    fig2 = plt.figure()
    sb.histplot(data=data, x="Care of Magical Creatures",
                hue=hues, legend=True, bins=50)
    fig2.savefig('../plots/histogram/answer.png')


if __name__ == "__main__":
    main()
