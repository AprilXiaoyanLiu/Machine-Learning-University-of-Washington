# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:49:24 2016

@author: April
"""

import pandas as pd
import numpy as np

house_data = pd.read_csv("{filepath}/kc_house_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
train_data = pd.read_csv("{filepath}kc_house_train_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test_data = pd.read_csv("{filepath}kc_house_test_data.csv",dtype = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})


# use the closed form solution from lecture to calculate the slope and intercept
def simple_linear_regression(input_feature,output):
    numerator = (input_feature * output).mean(axis=0) - (output.mean(axis=0))*(input_feature.mean(axis=0))
    denominator = (input_feature**2).mean(axis=0) - input_feature.mean(axis=0) * input_feature.mean(axis=0)
    slope = numerator/denominator
    intercept = output.mean(axis=0) - slope * (input_feature.mean(axis=0))
    return (intercept, slope)
    

sqft_living = train_data['sqft_living']
sqft_living_list = [i for i in train_data['sqft_living']]
sqft_living_array = np.array(sqft_living_list)


price_list = [m for m in train_data['price']]
price_list_array = np.array(price_list)


intercept_train,slope_train = simple_linear_regression(sqft_living_array, price_list_array)


def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = intercept + input_feature * slope
    return(predicted_output)
 
# use function to calcuate the estimated slope and intercept on the training data to predict 'price'given 'sqft_living'   
input_feature = 2650
print get_regression_predictions(2650, intercept_train, slope_train)

# What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data
def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    RSS = (((intercept + input_feature*slope) - output)**2).sum(axis=0)
    return(RSS)
    
print get_residual_sum_of_squares(sqft_living_array,price_list_array,intercept_train,slope_train)


# what is the estimated square-feet for a house costing $800,000?
def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output - intercept)/slope
    return(estimated_input)
    
output = 800000
print inverse_regression_predictions(output,intercept_train,slope_train)

# Which model (square feet or bedrooms) has lowest RSS on TEST data? 
sqft_living_array_test = np.array([a for a in test_data['sqft_living']])
bedrooms_array_test = np.array([b for b in test_data['bedrooms']])
price_array_test = np.array([c for c in test_data['price']])
intercept_sqf,slope_sqf = simple_linear_regression(sqft_living_array_test,price_array_test)
intercept_sqf

intercept_br, slope_br = simple_linear_regression(bedrooms_array_test,price_array_test)
RSS_sqf = get_residual_sum_of_squares(sqft_living_array_test,price_array_test,intercept_sqf,slope_sqf)
RSS_br = get_residual_sum_of_squares(bedrooms_array_test,price_array_test,intercept_br,slope_br)
print RSS_sqf - RSS_br
    

