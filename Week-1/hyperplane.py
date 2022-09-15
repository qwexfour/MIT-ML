#!/usr/bin/python3
"""Tools for hyperplanes.

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

def substitute(x, th, th0):
  return np.dot(np.transpose(th), x) + np.transpose(th0)

def signed_dist(x, th, th0):
  res = substitute(x, th, th0)
  return res / np.linalg.norm(th)

def positive(x, th, th0):
  return np.sign(substitute(x, th, th0))

def score(data, labels, th, th0):
  comp = (positive(data, th, th0) == labels)
  return np.sum(comp, axis=1)

def best_separator(data, labels, ths, th0s):
  """Selects the best separator.

  Parameters:
    data - a d by n array of floats (representing n data points in d
           dimensions);
    labels - a 1 by n array of elements in (+1, -1), representing target labels;
    ths - a d by m array of floats representing m candidate thetas (each theta
          has dimension d by 1);
    th0s - a 1 by m array of the corresponding m candidate theta_0s.

  Returns tuple of normal in the form of a d by 1 array and an offset in the
  form of 1 by 1 array.
  """
  idx = np.argmax(score(data, labels, ths, th0s))
  return (ths[:,idx:idx+1], th0s[:,idx:idx+1])

def _dumb_test():
  data = np.array([[ 1.0, 0.0, -1.5],
                   [-1.0, 1.0, -1.0]])
  labels = np.array([[1, -1, 1]])
  ths = np.array([[0.0,  1.0,  0.0],
                  [1.0, -1.0, -1.0]])
  th0s = np.array([[0.5, 0.0, -0.01]])

  expected_score = np.array([0, 2, 3])
  test_score = score(data, labels, ths, th0s)
  print(test_score)
  if (test_score != expected_score).any():
    print("Wrong score.")
    print("Expected score:")
    print(expected_score)

  expected_separator = (np.array([[0.0],
                                  [-1.0]]),
                        np.array([[-0.01]]))
  test_separator = best_separator(data, labels, ths, th0s)
  print(test_separator)
  if (test_separator[0] != expected_separator[0]).any() or \
     (test_separator[1] != expected_separator[1]).any():
    print("Wrong separator.")
    print("Expected separator:")
    print(expected_separator)

if __name__ == "__main__":
  _dumb_test()
