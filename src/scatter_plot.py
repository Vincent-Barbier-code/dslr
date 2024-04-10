import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main() -> None:
	data = pd.read_csv('../datasets/dataset_train.csv')

	hues = data["Hogwarts House"]

	data = data.drop(['Index', 'First Name', 'Last Name',
						'Birthday', 'Best Hand', 'Hogwarts House'], axis=1)

	plt.xticks([])
	plt.yticks([])

	if not os.path.exists('../plots/scatter'):
		os.makedirs('../plots/scatter')

	for f, course in enumerate(data.columns.values):
		plt.figure(f)
		fig, axes = plt.subplots(4, 3)
		plt.subplots_adjust(hspace=0.4, wspace=0.4, top=0.945)
		fig.set_figwidth(10)
		fig.set_figheight(10)
		i = 0
		for col in data.columns.values:
			if col == course:
				continue
			sb.scatterplot(data=data, x=course, y=col, hue=hues, ax=axes[i // 3, i % 3], legend=False)
			i += 1
		fig.savefig('../plots/scatter/' + course + '.png')
		print(f"Saving {course}...")
		plt.close(f)

	fig = plt.figure()
	sb.scatterplot(data=data, x="Defense Against the Dark Arts", y="Astronomy", hue=hues, legend=True)
	fig.savefig('../plots/scatter/answer.png')
	print(f"Saving answer...")

if __name__ == "__main__":
	main()
