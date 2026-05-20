import pandas as pd
import numpy as np

car_fuel_efficiency = pd.read_csv('car_fuel_efficiency.csv')

print(f'length of dataset is {car_fuel_efficiency.shape[0]}')

print(f'The number of fuel types in the dataset is {car_fuel_efficiency['fuel_type'].nunique()}')

print(f'The number of columns with missing values is {(car_fuel_efficiency.isnull().sum() > 0).sum()}')

asia_cars = car_fuel_efficiency[car_fuel_efficiency['origin'] == 'Asia']

print(f'The max fuel efficiency for Asian Cars is {asia_cars['fuel_efficiency_mpg'].max()}')

print(f'The Median value of the horsepower columns is {car_fuel_efficiency['horsepower'].median()}')

print(f'The Most frequent value on the horsepower column is {car_fuel_efficiency['horsepower'].mode()[0]}')

horsepower_mode = car_fuel_efficiency['horsepower'].mode()[0]

car_fuel_efficiency['horsepower'] = car_fuel_efficiency['horsepower'].fillna(horsepower_mode)

print(f'The Median value of the horsepower columns is {car_fuel_efficiency['horsepower'].median()}')

asia_cars = car_fuel_efficiency[car_fuel_efficiency['origin'] == 'Asia']

asia_cars_selected = asia_cars[['vehicle_weight', 'model_year']]

asia_cars_7 = asia_cars_selected.head(7)

X = asia_cars_7.values

XTX = X.T.dot(X)

XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv.dot(X.T).dot(y)

print(w.sum())