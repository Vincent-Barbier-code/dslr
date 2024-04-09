import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def plot(data):
	if not os.path.exists('../plots'):
		os.makedirs('../plots')

	for i in range(len(data.columns)):
		for j in range(len(data.columns)):
			if i != j and i != 0 and j != 0:
				sb.scatterplot(data=data, x=data.columns[i], y=data.columns[j], hue="Hogwarts House")
				plt.xlabel(data.columns[i])
				plt.ylabel(data.columns[j])
				plt.title(data.columns[i] + " VS " + data.columns[j])
				plt.savefig('../plots/' + data.columns[i] + 'VS' + data.columns[j] + '.png')
				plt.clf()

