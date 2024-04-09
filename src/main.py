from extract import Extract
from plots.scatter_plot import plot

def main():
	data = Extract("datasets/dataset_train.csv").get_data()
	data = data.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1)
	print(data)
	plot(data)
 
if __name__ == "__main__":
	main()