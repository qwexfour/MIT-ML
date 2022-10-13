#!/usr/bin/python3
"""A module that implements perceptron algorithm.

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

def perceptron(data, labels, params={}, hook=None):
  """Perceptron algorithm implementation.

  Parameters:
    data - a numpy array of dimension d by n;
    labels - a numpy array of dimension 1 by n;
    params - a dictionary specifying extra parameters to this algorithm, this
             algorithm should run a number of iterations equal to T;
    hook - either None or a function that takes the tuple (th, th0) as an
           argument and displays the separator graphically.
  """
  # if T not in params, default to 100
  T = params.get('T', 100)
  through_origin = params.get('through_origin', False)

  # Your implementation here
  num_dims, data_size = data.shape
  theta = np.zeros((num_dims, 1))
  theta_0 = 0.0

  for t in range(T):
    for i in range(data_size):
      x = data[:, i:i+1]
      y = labels[0, i]
      prediction = np.dot(np.transpose(theta), x) + theta_0
      if y * prediction <= 0:
        theta += y * x
        if not through_origin:
          theta_0 += y
        if hook:
          hook((theta, theta_0))

  return (theta, theta_0)

def averaged_perceptron(data, labels, params={}, hook=None):
  """Averaged perceptron algorithm implementation.

  This algorithm is just like the standard pereceptron algorith except it
  returns the average of all the intermediate thetas and theta_0s.

  Parameters:
    data - a numpy array of dimension d by n;
    labels - a numpy array of dimension 1 by n;
    params - a dictionary specifying extra parameters to this algorithm, this
             algorithm should run a number of iterations equal to T;
    hook - either None or a function that takes the tuple (th, th0) as an
           argument and displays the separator graphically.
  """
  # if T not in params, default to 100
  T = params.get('T', 100)
  through_origin = params.get('through_origin', False)

  # Your implementation here
  num_dims, data_size = data.shape
  theta = np.zeros((num_dims, 1))
  theta_0 = 0.0
  theta_sum = np.zeros((num_dims, 1))
  theta_0_sum = 0.0

  for t in range(T):
    for i in range(data_size):
      x = data[:, i:i+1]
      y = labels[0, i]
      prediction = np.dot(np.transpose(theta), x) + theta_0
      if y * prediction <= 0:
        theta += y * x
        if not through_origin:
          theta_0 += y
        if hook:
          hook((theta, theta_0))
      theta_sum += theta
      theta_0_sum += theta_0

  sum_count = T * data_size
  return (theta_sum / sum_count, theta_0_sum / sum_count)
