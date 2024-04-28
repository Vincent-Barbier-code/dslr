import pandas as pd
import numpy as np
import math
import argparse


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
        return values[int(rank)]
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
    fields = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "unique",
        "freq",
        "range",
    ]
    for field in fields:
        statistics[field] = []

    for feature in dataframe.columns.values:
        count = 0
        total = 0
        values = dataframe[feature].values
        values = np.sort(values)
        occurences = dict()
        for value in values:
            if np.isnan(value):
                continue
            count += 1
            total += value
            if not value in occurences:
                occurences[value] = 1
            else:
                occurences[value] += 1
        try:
            statistics["count"].append(format_number(count))
            statistics["mean"].append(format_number(total / count))
            statistics["min"].append(format_number(values[0]))
            statistics["25%"].append(format_number(percentile(values, 25)))
            statistics["50%"].append(format_number(percentile(values, 50)))
            statistics["75%"].append(format_number(percentile(values, 75)))
            statistics["max"].append(format_number(values[-1]))
            statistics["unique"].append(format_number(len(occurences)))
            statistics["freq"].append(format_number((len(occurences) / count) * 100))
            statistics["range"].append(format_number(values[-1] - values[0]))

            sum = 0
            for value in values:
                if np.isnan(value):
                    continue
                sum += (value - (total / count)) ** 2
            statistics["std"].append(format_number(math.sqrt(sum / count)))
        except:
            exit("Unable to describe current dataset")


def format_number(value: float) -> str:
    """Format the value to align it to the right and keep 3 decimals

    Args:
                    value (float): value to format

    Returns:
                    str: formatted string
    """
    return "{:>12.3f}".format(value)


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
                    str: Metrics of dataframe
    """
    statistics = {}
    headers = []
    for header in dataframe.columns.values.tolist():
        headers.append("{:>12}".format(header[:7] + "." if len(header) > 7 else header))

    headers.insert(0, "")
    get_statistics(dataframe, statistics)

    data = [
        format_stats(statistics, "count"),
        format_stats(statistics, "unique"),
        format_stats(statistics, "freq"),
        format_stats(statistics, "range"),
        format_stats(statistics, "mean"),
        format_stats(statistics, "std"),
        format_stats(statistics, "min"),
        format_stats(statistics, "25%"),
        format_stats(statistics, "50%"),
        format_stats(statistics, "75%"),
        format_stats(statistics, "max"),
    ]
    out = ""

    col_width = max(len(word) for word in headers) - 3
    data.insert(0, headers)
    for row in data:
        out += ("".join(word.ljust(col_width) for word in row)) + "\n"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the dataset")
    args = parser.parse_args()

    data = pd.read_csv(args.path)
    data = data.select_dtypes(include=np.number)
    data.dropna(inplace=True)
    print(describe(data))
