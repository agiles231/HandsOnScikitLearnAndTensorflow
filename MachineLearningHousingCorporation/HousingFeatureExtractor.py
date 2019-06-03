import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class HousingFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_col_index, self.bedroom_col_index, self.population_col_index, self.households_col_index = 3,4,5,6
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_col_index]/X[:, self.households_col_index]
        population_per_household = X[:, self.population_col_index]/X[:, self.households_col_index]
        X_ = np.c_[X, rooms_per_household, population_per_household]
        if (self.add_bedrooms_per_room):
            bedrooms_per_room = X[:, self.bedroom_col_index]/X[:, self.rooms_col_index]
            X_ = np.c_[X_, bedrooms_per_room]
        return X_


        