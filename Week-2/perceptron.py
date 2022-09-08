#!/usr/bin/python3
""" Perceptron algorithm implementation.

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

def generate_input(num_dims, num_points, dist_func = lambda x : x):
  """Randomly generates an input for a linear separator.

  Parameters:
    num_dims - the number of dimensions;
    num_points - the number of points to produce;
    dist_func - distribution function, it takes real values in [0, 1) range or
                arrays of those values and returns a corresponding value with a
                different distribution; does not change the distribution by
                default.
  Result:
    Generated points;
    Labels for those points;
    theta and theta_0 that defines the chosen separator.
  """
  points = dist_func(np.random.rand(num_dims, num_points))

  # generating a plane parameters
  theta = np.random.rand(num_dims)
  point_on_plane = dist_func(np.random.rand(num_dims))
  theta_0 = -np.dot(theta, point_on_plane)

  labels = np.dot(np.transpose(theta), points) + theta_0
  labels = np.sign(labels)

  return (points, labels, theta, theta_0)


def main():
  points, labels, theta, theta_0 = generate_input(2, 15)
  print(points)
  print(labels)

if __name__ == "__main__":
  main()
