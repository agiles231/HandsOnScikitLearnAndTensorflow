import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import dataload as dl
from HousingFeatureExtractor import HousingFeatureExtractor
from HousingDataAttrSelector import HousingDataAttrSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

def main():

    housing = dl.load_housing_data()
    housing.dropna(subset=["total_bedrooms"], how='all', inplace=True)
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.iloc[train_index]
        strat_test_set = housing.iloc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    
    housing_cat = housing["ocean_proximity"]
    #housing_cat_encoded, housing_categories = housing_cat.factorize()
    #encoder = OneHotEncoder(categories='auto')
    #housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ("selector", HousingDataAttrSelector(num_attribs)),
        ("imputer", SimpleImputer(strategy="median")),
        ("featureExtractor", HousingFeatureExtractor()),
        ("std_scalar", StandardScaler())
    ])

    #cat_pipeline = Pipeline([
    #    ("selector", HousingDataAttrSelector(cat_attribs)),
    #    ("cat_encoder", CategoricalEncoder(encoding="onehot-dense"))
    #])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])
    housing_prepared = full_pipeline.fit_transform(housing)
    #print(housing_prepared.loc[19391])
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    housing_predictions = lin_reg.predict(housing_prepared)
    print("Predictions: ", housing_predictions)
    print("Labels: ", some_labels)

    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)



if __name__ == "__main__":
    main()