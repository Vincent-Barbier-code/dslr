import pandas as pd
import numpy as np
from columnar import columnar
import math


def percentile(values: np.ndarray, p: float) -> float:
	"""Get Pth percentile

	Args:
		values (np.ndarray): sorted data
		p (float): percentile wanted

	Returns:
		float: value of pth percentile
	"""

	rank = (p / 100) * (len(values) + 1)
	if rank.is_integer():
		return (values[int(rank)])
	f_rank = int(math.floor(rank))
	c_rank = int(math.ceil(rank))
	f_value = values[f_rank]
	c_value = values[c_rank]
	return f_value + (rank - f_rank) * (c_value - f_value)

def get_statistics(dataframe: pd.DataFrame, statistics: dict) -> None:
	"""Compute the statistics and saves them in the dictionary passed as a parameter

	Args:
		dataframe (pd.DataFrame): dataframe
		statistics (dict): dictionnary used to store stats
	"""
	fields = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
	for field in fields:
		statistics[field] = []

	for feature in dataframe.columns.values:
		count = 0
		total = 0
		values = dataframe[feature].values
		values = np.sort(values)
		for value in values:
			if np.isnan(value):
				continue
			count += 1
			total += value
		statistics["count"].append(format_number(count))
		statistics["mean"].append(format_number(total / count))
		statistics["min"].append(format_number(values[0]))
		statistics["25%"].append(format_number(percentile(values, 25)))
		statistics["50%"].append(format_number(percentile(values, 50)))
		statistics["75%"].append(format_number(percentile(values, 75)))
		statistics["max"].append(format_number(values[-1]))

		sum = 0
		for value in values:
			if np.isnan(value):
				continue
			sum += (value - (total / count)) ** 2
		statistics["std"].append(format_number(math.sqrt(sum / count)))

def format_number(value: float) -> str:
	"""Format the value to align it to the right and keep 3 decimals

	Args:
		value (float): value to format

	Returns:
		str: formatted string
	"""
	return "{:>10.3f}".format(value)

def format_stats(statistics: dict, field: str) -> list:
	"""Format certain statistics for columnar

	Args:
		statistics (dict): dictionnary of all stats
		field (str): target stats

	Returns:
		list: Correct format for columnar
	"""
	return [field] + statistics[field]

def describe(dataframe: pd.DataFrame) -> str:
	"""Describe a dataframe

	Args:
		dataframe (pd.DataFrame): dataframe to describe

	Returns:
		str: Representation of dataframe
	"""
	statistics = {}
	headers = []
	for header in dataframe.columns.values.tolist():
		headers.append("{:>10}".format(header[:9] + '.' if len(header) > 9 else header))


	headers.insert(0, "{:<3}".format(''))
	get_statistics(dataframe, statistics)
	data = [
			format_stats(statistics, "count"),
			format_stats(statistics, "mean"),
			format_stats(statistics, "std"),
			format_stats(statistics, "min"),
			format_stats(statistics, "25%"),
			format_stats(statistics, "50%"),
			format_stats(statistics, "75%"),
			format_stats(statistics, "max"),
			]
	out = columnar(data, headers=headers, no_borders=True, preformatted_headers=True)
	return (out)

if __name__ == "__main__":
	data = pd.read_csv('../datasets/dataset_train.csv')
	data = data.select_dtypes(include=np.number)
	data.dropna(inplace=True)
	print(describe(data))
	