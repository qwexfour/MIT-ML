#!/usr/bin/python3
"""Support vector machine model implementation.

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

import sys
import numpy as np

# FIXME: find a better way.
sys.path.append('../Week-1')
import hyperplane as hp

def hinge(v):
  """Hinge function implementation"""
  return np.maximum(0, 1 - v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
  """
  Calculates hinge losses for labeled points and hyperplanes.

  NOTE: Gives the same result as margin.hinge_loss with the reference margin
  equal to 1 / norm(th).

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - m hyperplane theta parameters ([d x m] numpy array of numbers);
    th0 - m hyperplane theta_0 parameters ([1 x m] numpy array of numbers,
          or a single number for m == 1);

  Returns [m x n] matrix of hinge losses for every plane and data combination.
  """
  v = y * hp.substitute(x, th, th0)
  return hinge(v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
  """
  Calculates SVM objective.

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - hyperplane theta parameter ([d x 1] numpy array of numbers);
    th0 - hyperplane theta_0 parameter ([1 x 1] numpy array of numbers,
          or a single number);

  Returns [m x n] matrix of hinge losses for every plane and data combination.
  """
  return np.mean(hinge_loss(x, y, th, th0)) + lam * np.dot(th.T, th)[0, 0]

def _super_simple_separable_svm_obj_test():
  x_1 = np.array([[2, 3, 9, 12],
                [5, 2, 6, 5]])
  y_1 = np.array([[1, -1, 1, -1]])

  th1, th1_0 = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

  # Test case 1
  ans = svm_obj(x_1, y_1, th1, th1_0, .1)
  print(ans)

  # Test case 2
  ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
  print(ans)

def _main():
  pass

if __name__ == "__main__":
  _main()
