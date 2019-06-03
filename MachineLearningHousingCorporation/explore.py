import os
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import dataClean as dc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    #fetch_housing_data() # we have already done this before, and we don't want to change our data
    housing_data = dc.load_housing_data()
    housing_data = dc.clean_housing_data(housing_data)
    train_data, test_data = train_test_split(housing_data, test_size = 0.2, random_state=42)
    housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
    housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)
    #housing_data["income_cat"].hist()
    #plt.show()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
        strat_train_set = housing_data.loc[train_index]
        strat_test_set = housing_data.loc[test_index]
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    housing_data = strat_train_set.copy()
    #housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    #plt.show()

    #housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4
    #    , s=housing_data["population"]/100, label="population", figsize=(10,7)
    #    , c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    #plt.show()
    #plt.legend()
    corr_matrix = housing_data.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    #scatter_matrix(housing_data[attributes], figsize=(12,8))
    #plt.show()
    housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()
    housing_data["rooms_per_household"] = housing_data["total_rooms"]/housing_data["households"]
    housing_data["bedrooms_per_room"] = housing_data["total_bedrooms"]/housing_data["total_rooms"]
    housing_data["population_per_household"] = housing_data["population"]/housing_data["households"]
    corr_matrix = housing_data.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

if __name__ == "__main__":
    main()