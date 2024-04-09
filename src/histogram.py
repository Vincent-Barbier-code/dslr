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
	plt.xticks(visible=False)
	plt.subplots_adjust(bottom=0.064, top=0.945, hspace=0.67)

	for i, col in enumerate(data.columns.values):
		sb.histplot(ax=axes[i // 3, i % 3], data=data, x=col, hue=hues, legend=False, bins=50)

	fig.savefig('../plots/histogram/all.png')
	fig2 = plt.figure()
	sb.histplot(data=data, x="Care of Magical Creatures", hue=hues, legend=True, bins=50)
	fig2.savefig('../plots/histogram/answer.png')
	plt.show()

if __name__ == "__main__":
	main()