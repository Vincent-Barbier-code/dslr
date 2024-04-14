import pandas as pd
import sys


class Extract:
    def __init__(self, csv_file, header=None):
        self.data = None
        try:
            if header:
                self.data = pd.read_csv(csv_file, header=header)
            else:
                self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            sys.exit(f"File not found {csv_file}")
        except pd.errors.EmptyDataError:
            sys.exit("Empty file")

    def standardize(self) -> pd.DataFrame:
        self.data = (self.data - self.data.mean()) / self.data.std()
        return self.data

    def fillNaN(self, exlusions=[]) -> pd.DataFrame:
        keys = set(self.data.columns.values).difference(exlusions)
        for key in keys:
            self.data.fillna(value={key: self.data[key].mean()}, inplace=True)
        return self.data

    def dropColumns(self, col: list[str]) -> pd.DataFrame:
        self.data = self.data.drop(col, axis=1)
        return self.data

    def get_data_training(
        self, label_name: str, need_facto: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = self.data.drop(label_name, axis=1)
        y = None
        if need_facto:
            y = pd.factorize(self.data[label_name])[0]
        else:
            y = self.data[label_name]
        x = (x - x.mean()) / x.std()
        return x, y

    def get_data_test(self) -> pd.DataFrame:
        x = self.data
        x = (x - x.mean()) / x.std()
        return x
