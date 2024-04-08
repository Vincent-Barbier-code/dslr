import pandas as pd

class Extract:
	def __init__(self, csv_file, header=None):
		try:
			data = pd.read_csv(csv_file, header=header)
			self.data = data
		except FileNotFoundError:
			print("File not found")
			self.data = None
		except pd.errors.EmptyDataError:
			print("Empty file")
			self.data = None

	def get_data(self):
		return self.data

	def standardize(self):
		self.data = (self.data - self.data.mean()) / self.data.std()
		return self.data