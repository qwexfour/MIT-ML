#!/usr/bin/python3
"""Tools for calculating margin.

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
import hyperplane as hplane

def margin(data, labels, thetas, theta_0s):
  """
  Calculates margin characteristics for labeled points and hyperplanes.

  Parameters:
    data - n data points in d dimensions ([d x n] numpy array of numbers);
    labels - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
             single number +1 or -1 for n == 1);
    thetas - m hyperplane theta parameters ([d x m] numpy array of numbers);
    theta_0s - m hyperplane theta_0 parameters ([1 x m] numpy array of numbers,
               or a single number for m == 1).

  Returns [m x n] matrix of margins computed for all combinations of data points
  and hyperplanes.
  """
  return labels * hplane.signed_dist(data, thetas, theta_0s)

def margin_features(data, labels, thetas, theta_0s):
  """
  Calculates margin characteristics for labeled points and hyperplanes.

  Parameters:
    data - n data points in d dimensions ([d x n] numpy array of numbers);
    labels - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
             single number +1 or -1 for n == 1);
    thetas - m hyperplane theta parameters ([d x m] numpy array of numbers);
    theta_0s - m hyperplane theta_0 parameters ([1 x m] numpy array of numbers,
               or a single number for m == 1).

  Returns [m x 3] matrix of margins characteristics. The characteristics are:
    sum of margins,
    minimal margin,
    maximal margin.
  """
  margins = margin(data, labels, thetas, theta_0s)
  m_sum = np.sum(margins, axis=1, keepdims=True)
  m_min = np.amin(margins, axis=1, keepdims=True)
  m_max = np.amax(margins, axis=1, keepdims=True)
  return np.hstack((m_sum, m_min, m_max))

def _task_1():
  """Implements the first task."""
  data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                   [1, 1, 2, 2,  2,  2,  2, 2]])
  labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
  blue_th = np.array([[0, 1]]).T
  blue_th0 = -1.5
  red_th = np.array([[1, 0]]).T
  red_th0 = -2.5

  ths = np.hstack((red_th, blue_th))
  th0s = np.array([[red_th0, blue_th0]])

  res = margin_features(data, labels, ths, th0s)
  print("Margin characteristics for red and blue separators from the first "
        "task.\nFirst row is for the red separator, second - for the blue "
        "one.\nColumns are: Ssum, Smin, Smax.")
  print(res)

def _single_sep_test():
  data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                   [1, 1, 2, 2,  2,  2,  2, 2]])
  labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
  red_th = np.array([[1, 0]]).T
  red_th0 = -2.5

  res = margin_features(data, labels, red_th, red_th0)
  print("Testing that margin_features function works fine with a single "
        "separator:")
  print(res)

def _degenerate_case_test():
  data = np.array([[10],
                   [2]])
  labels = 1
  red_th = np.array([[1, 0]]).T
  red_th0 = -2.5

  res = margin_features(data, labels, red_th, red_th0)
  print("Testing that margin_features function works fine for degenerate case with a single "
        "separator and a single data point:")
  print(res)

def _main():
  _task_1()

if __name__ == "__main__":
  _main()
