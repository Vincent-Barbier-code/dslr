import pandas as pd

class Extract:
	def __init__(self, csv_file):
		try:
			data = pd.read_csv(csv_file, header=0)
			self.data = data
		except FileNotFoundError:
			print("File not found")
			self.data = None
		except pd.errors.EmptyDataError:
			print("Empty file")
			self.data = None

	def get_data(self):
		return self.data

	def standardize(self, i=0, y=0):
		try:
			if y == 0:
				self.data = self.data.iloc[:, i:]
			else:
				self.data = self.data.iloc[:, i:y]
		except IndexError:
			print("Index out of range")
			return self.data
		if self.data is not int or self.data is not float:
			self.data = self.data.apply(pd.to_numeric, errors='coerce')
		self.data = (self.data - self.data.mean()) / self.data.std()
		return self.data