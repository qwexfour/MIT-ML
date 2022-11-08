#!/usr/bin/python3
"""Square loss utilities model implementation.

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

def square_loss(x, y, th, th0):
  """
  Calculates square losses for the provided data and hyperplanes.

  Parameters:
    x - n input points in d dimensions from the training data set ([d x n]
        numpy array of numbers);
    y - n output values from the training data set ([1 x n] numpy array of 
        numbers, or a single number for n == 1);
    th - m hyperplane theta parameters ([d x m] numpy array of numbers);
    th0 - m hyperplane theta_0 parameters ([1 x m] numpy array of numbers,
          or a single number for m == 1);

  Returns [m x n] matrix of square losses for every plane and data combination.
  """
  guess = hp.substitute(x, th, th0)
  diff = y - guess
  return diff**2
