#!/usr/bin/python3
"""Tools for calculating gradient descent.

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

def gd(f, df, x0, step_size_fn, max_iter):
  """ Calculates gradient descent.

  Parameters:
    f - a function whose input is an x, a column vector, and returns a scalar;
    df - a function whose input is an x, a column vector, and returns a column
         vector representing the gradient of f at x;
    x0 - an initial value of x, x0, which is a column vector;
    step_size_fn - a function that is given the iteration index (an integer)
                   and returns a step size;
    max_iter - the number of iterations to perform.

  Returns a tuple that consists of:
    x - the value at the final step
    fs - the list of values of f found during all the iterations
         (including f(x0))
    xs - the list of values of x found during all the iterations (including x0)
  """
  x = x0
  fs = [f(x0)]
  xs = [x0]
  for i in range(max_iter):
    x = x - step_size_fn(i) * df(x)
    fs.append(f(x))
    xs.append(x)
  return (x, fs, xs)

def part_deriv(f, x, idx, delta=0.001):
  """ Numerically calculates partial derivative df / dx_idx.

  Parameters:
    f - a function which derivative must be taken, whose input is an x,
        a column vector, and returns a scalar;
    x - a point at which the derivative must be taken, a column vector;
    idx - along which dimension the derivative must be taken;
    delta - half the distance between 2 points used to calculate the derivative,
            a number.

  The value of the partial derivative is returned.
  """
  vec_delta = np.zeros(x.shape)
  vec_delta[idx, 0] = delta
  return (f(x + vec_delta) - f(x - vec_delta)) / (2 * delta)

def num_grad(f, delta=0.001):
  """ Numerically calculates gradient function.

  Parameters:
    f - a function which gradient must be taken, whose input is an x,
        a column vector, and returns a scalar;
    delta - half the distance between 2 points used to calculate the gradient,
            a number.

  The function that calculates the gradient of the provided function is
  returned.
  """
  def df(x):
    return np.array([[part_deriv(f, x, i, delta) for i in range(x.shape[0])]]).T
  return df
