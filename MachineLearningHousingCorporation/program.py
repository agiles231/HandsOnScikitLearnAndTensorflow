import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH) :
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def clean_housing_data(housing_data):
    return housing_data


def main():
    #fetch_housing_data() # we have already done this before, and we don't want to change our data
    housing_data = load_housing_data()
    housing_data = clean_housing_data(housing_data)
    train_data, test_data = train_test_split(housing_data, test_size = 0.2, random_state=42)
    housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
    housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)
    housing_data["income_cat"].hist()
    plt.show()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
        strat_train_set = housing_data.loc[train_index]
        strat_test_set = housing_data.loc[test_index]
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    housing_data = strat_train_set.copy()
    housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

    housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4
        , s=housing_data["population"]/100, label="population", figsize=(10,7)
        , c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()
    plt.legend()
    
    

if __name__ == "__main__":
    main()