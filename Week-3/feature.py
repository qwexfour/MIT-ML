#!/usr/bin/python3
"""A module that implements feature tools.

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

def one_hot(x, k):
  """ Produces "one hot" feature representation.

  Parameters:
    x - linear feature representation, an integer from [1, k] range;
    k - feature set cardinality.

  Returns [k x 1] numpy array with "one hot" representation of the feature.
  """
  vec = np.zeros((k, 1))
  vec[x - 1, 0] = 1
  return vec
