# pyrefly: ignore [missing-import]
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
 
housing = pd.read_csv('housing.csv')

housing['income_cat'] = pd.cut(housing["median_income"], 
bins = [0, 1.5, 3, 4.5, 6, np.inf], 
labels= [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strait_train_set = housing.loc[train_index].drop('income_cat', axis= 1)
    strait_test_set = housing.loc[test_index].drop('income_cat', axis= 1)

housing = strait_train_set.copy()

housing_lables = strait_train_set['median_house_value'].copy()
housing = strait_train_set.drop('median_house_value', axis=1)

# print(housing)
# print(housing_lables)

num_attributes = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scalar", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

housin_prepared = full_pipeline.fit_transform(housing)
# print(housin_prepared)

lin_reg = LinearRegression()
lin_reg.fit(housin_prepared, housing_lables)
lin_pred = lin_reg.predict(housin_prepared)
lin_rmse = root_mean_squared_error(housing_lables, lin_pred)
print(f"The root mean squared error for Linear Regression is {lin_rmse}")

dec_reg = DecisionTreeRegressor()
dec_reg.fit(housin_prepared, housing_lables)
dec_pred = dec_reg.predict(housin_prepared)
dec_rmse = root_mean_squared_error(housing_lables, dec_pred)
print(f"The root mean squared error for Decision Regression is {dec_rmse}")

rand_reg = RandomForestRegressor()
rand_reg.fit(housin_prepared, housing_lables)
rand_pred = rand_reg.predict(housin_prepared)
rand_rmse = root_mean_squared_error(housing_lables, rand_pred)
print(f"The root mean squared error for Random Forest Regression is {rand_rmse}")

# lin_reg = LinearRegression()
# lin_reg.fit(housin_prepared, housing_lables)
# lin_pred = lin_reg.predict(housin_prepared)
# lin_rmse = root_mean_squared_error(housing_lables, lin_pred)
# print(lin_rmse)