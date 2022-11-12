#!/usr/bin/python3
"""Linear regression over the auto data set.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import code_for_hw5 as hw5

def lambdas(polynomial_order):
  assert polynomial_order > 0 and polynomial_order < 4, "wrong polynomial order"
  if polynomial_order == 3:
    return range(0, 201, 20)
  return [x / 100 for x in range(0, 11)]
#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw5.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw5.standard and hw5.one_hot.
# 'name' is not numeric and would need a different encoding.
features1 = [('cylinders', hw5.standard),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

features2 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = hw5.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = hw5.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = hw5.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

#Your code for cross-validation goes here
#Make sure to scale the RMSE values returned by xval_learning_alg by sigma,
#as mentioned in the lab, in order to get accurate RMSE values on the dataset

min_score = float('inf')
for feature_idx, data in enumerate(auto_data):
  for polynomial_order in range(1, 4):
    polynomial_transformation = \
      hw5.make_polynomial_feature_fun(polynomial_order)
    for lam in lambdas(polynomial_order):
      transformed_data = polynomial_transformation(data)
      score = hw5.xval_learning_alg(transformed_data, auto_values, lam, 10)
      if score < min_score:
        best_conf = feature_idx + 1, polynomial_order, lam
        min_score = score

print("The best configuration is: ", best_conf,
      " (feature set, polynomial order, lambda).")
print("Root Mean Square Error (RMSE) is ", min_score * sigma, " mpg.")

min_score = float('inf')
polynomial_transformation = hw5.make_polynomial_feature_fun(3)
for lam in lambdas(3):
  transformed_data = polynomial_transformation(auto_data[0])
  score = hw5.xval_learning_alg(transformed_data, auto_values, lam, 10)
  if score < min_score:
    best_lam = lam
    min_score = score

print("Considering 3rd polynomial order over the 1st feature set, the best ",
      "lambda and the RMSE are: ", best_lam, min_score * sigma, ".")
