#!/usr/bin/python3
"""Tools for linearly separable data.

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
import matplotlib.pyplot as plt

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
    Generated points (numpy [num_dims x num_points] array);
    Labels for those points (numpy [1 x num_points] array);
    theta (numpy [num_dims x 1] array) and theta_0 (numpy [1 x 1]) array that
    define the chosen separator.
  """
  points = dist_func(np.random.rand(num_dims, num_points))

  # generating a plane parameters
  theta = np.random.rand(num_dims) - np.array([0.5] * num_dims)
  point_on_plane = dist_func(np.random.rand(num_dims))
  theta_0 = -np.dot(theta, point_on_plane)
  theta = np.transpose(np.array([theta]))

  labels = np.dot(np.transpose(theta), points) + theta_0
  labels = np.sign(labels)

  return (points, labels, theta, theta_0)

def draw(points, labels, equation):
  """Draws the provided points and separator.

  Parameters:
    points - 2 x N matrix, only 2D case is supported;
    labels - 1 x N matrix, labels for the provided points: +1 or -1;
    equation - a tuple of equation parameters: (theta, theta_0).
  """
  axes = plt.subplot(111)
  axes.set_aspect('equal')

  _draw_points(points, labels)
  min_point, max_point, epsilon = _calculate_points_properties(points)
  _draw_line(equation, min_point, max_point, epsilon)
  plt.show()

def _calculate_points_properties(points):
  """Calculates some properties of the provided points.

  Parameters:
    points - 2 x N matrix, only 2D case is supported;
  Result:
    min_point, max_point - diagonal points that define a "box" that contain all
                           the points;
                           ^   |------max
                           |   |       |
                           |   |       |
                           |  min------|
                           |------------->
    epsilon - allowed linear calculation error: if the distance between 2 points
              is less than epsilon, they are equal.
  """
  min_point = np.amin(points, axis=1, keepdims=True)
  max_point = np.amax(points, axis=1, keepdims=True)

  num_points = points.shape[1]
  box_size = max_point - min_point
  area_per_point = box_size[0, 0] * box_size[1, 0] / num_points
  # Allowed error is a fraction of the average linear space occupied by a point.
  epsilon = np.sqrt(area_per_point) / 1000

  return (min_point, max_point, epsilon)

def _draw_points(points, labels):
  """Draws the provided points and separator.

  Parameters:
    points - 2 x N matrix, only 2D case is supported;
    labels - 1 x N matrix, labels for the provided points: +1 or -1;
  Positive points are displayed as "+", negative - as "-".
  """
  labelled_points = np.concatenate((points, labels), axis=0)
  pos_points = labelled_points[0:-1, labelled_points[-1, :] > 0]
  neg_points = labelled_points[0:-1, labelled_points[-1, :] <= 0]

  plt.plot(pos_points[0], pos_points[1], 'k+')
  plt.plot(neg_points[0], neg_points[1], 'k_')

def _draw_line(equation, min_point, max_point, epsilon):
  """Draw the line defined by the equation.

  Parameters:
    equation - a tuple of equation parameters: (theta, theta_0).
    min_point, max_point - diagonal points that define a "box" in which the line
                           should be drawn;
    epsilon - allowed linear calculation error: if the distance between 2 points
              is less than epsilon, they are equal.
  The algorithm defines 2 points where the provided line crosses the borders of
  the box in which it should be drawn. The line is drawn between those 2 points.
  There is an augment in the middle of the line that shows the direction of the
  normal.
  """
  line_points = _define_box_crossing(equation, min_point, max_point, epsilon)
  medium = (line_points[:, 0:1] + line_points[:, 1:2]) / 2
  scaled_theta = _scale_vector_for_box(equation[0], max_point - min_point)
  theta_tip = medium + scaled_theta
  line_points = np.concatenate((line_points[:, 0:1], medium, theta_tip, medium,
                                line_points[:, 1:2]), axis=1)
  plt.plot(line_points[0], line_points[1], 'b-')

def _scale_vector_for_box(vec, box_size, factor=10):
  """Scales the vector to match the box size.

  Parameters:
    vec - the provided column vector to be scaled;
    box_size - a column vector that defines the box size;
    factor - what fraction of the box diagonal should the scaled vector be
             equal to (norm(scaled_vec) == norm(box_size) / factor).
  The scaled vector is returned.
  """
  desired_norm = np.linalg.norm(box_size) / factor
  current_norm = np.linalg.norm(vec)
  return vec * (desired_norm / current_norm)

def _define_box_crossing(equation, min_point, max_point, epsilon):
  """Defines where the line crosses the box.

  Parameters:
    equation - a tuple of equation parameters: (theta, theta_0).
    min_point, max_point - diagonal points that define a "box";
                           ^   |------max
                           |   |       |
                           |   |       |
                           |  min------|
                           |------------->
    epsilon - allowed linear calculation error: if the distance between 2 points
              is less than epsilon, they are equal.
  The algorithm defines 2 points where the provided line crosses the borders of
  the box. The horizontal and vertical degenerated cases considered at first and
  separately. The main part of algorithm goes through the 4 box borders and
  finds which of them are crossed by the line.
  """
  theta, theta_0 = equation
  box_size = max_point - min_point

  # Horizontal and vertical degenerate cases.
  for i in range(2):
    if _has_zero_coordinate(theta, box_size, epsilon, zero_axis=i):
      return _define_degenerate_box_crossin(equation, min_point, max_point,
                                            epsilon, zero_axis=i)

  result = np.array([[], []])

  # FIXME: Consider a case whith crossing in the corner. The current algorithm
  #        will add the same point twice in this case.
  # Left (minimal) vertical (along x2) border:
  x = min_point[0, 0]
  candidate = np.array([[x],
                        [_get_other_coordinate(equation, x, given_axis=0)]])
  if candidate[1, 0] > min_point[1, 0] - epsilon and \
     candidate[1, 0] < max_point[1, 0] + epsilon:
     result = np.concatenate((result, candidate), axis=1)

  # Top (maximal) horizontal (along x1) border:
  x = max_point[1, 0]
  candidate = np.array([[_get_other_coordinate(equation, x, given_axis=1)],
                        [x]])
  if candidate[0, 0] > min_point[0, 0] - epsilon and \
     candidate[0, 0] < max_point[0, 0] + epsilon:
     result = np.concatenate((result, candidate), axis=1)

  # Right (maximal) vertical (along x2) border:
  x = max_point[0, 0]
  candidate = np.array([[x],
                        [_get_other_coordinate(equation, x, given_axis=0)]])
  if candidate[1, 0] > min_point[1, 0] - epsilon and \
     candidate[1, 0] < max_point[1, 0] + epsilon:
     result = np.concatenate((result, candidate), axis=1)

  # Bottom (minimal) horizontal (along x1) border:
  x = min_point[1, 0]
  candidate = np.array([[_get_other_coordinate(equation, x, given_axis=1)],
                        [x]])
  if candidate[0, 0] > min_point[0, 0] - epsilon and \
     candidate[0, 0] < max_point[0, 0] + epsilon:
     result = np.concatenate((result, candidate), axis=1)

  assert result.shape[1] == 2, "Must have found only 2 intersections"
  return result

def _get_other_coordinate(equation, x, given_axis):
  """Given a coordinate of a point on the line calculates the other coordinate.

  Parameters:
    equation - a tuple of equation parameters: (theta, theta_0);
    x - the given coordinate;
    given_axis - the axis of the given coordinate.
  Only 2D case is considered.
  The other coordinate is returned.
  """
  assert given_axis == 0 or given_axis == 1, "only 2D case is considered"
  other_axis = 1 - given_axis
  theta, theta_0 = equation
  return -(theta[given_axis, 0] * x + theta_0) / theta[other_axis, 0]

def _has_zero_coordinate(point, box_size, epsilon, zero_axis):
  """Checks that one of point's coordinates is close to zero.

  Parameters:
    point - 2D point represented as a column vector;
    box_size - size of the viewed box;
    epsilon - allowed linear calculation error: if the distance between 2 points
              is less than epsilon, they are equal;
    zero_axis - the axis along which the point is expected to have zero
                coordinate.
  Box size with epsilon give understanding of the minimal distinguishable angle.
  """
  assert zero_axis == 0 or zero_axis == 1, "only 2D case is considered"
  other_axis = 1 - zero_axis
  if point[other_axis, 0] == 0:
    return False
  point_ratio = abs(point[zero_axis, 0] / point[other_axis, 0])
  epsilon_ratio = epsilon / box_size[other_axis, 0]
  assert epsilon_ratio > 0
  return point_ratio < epsilon_ratio

def _define_degenerate_box_crossin(equation, min_point, max_point, epsilon,
                                   zero_axis):
  """Defines points of the line and the box intersection for the corner case.

  Parameters:
    equation - a tuple of equation parameters: (theta, theta_0);
    min_point, max_point - diagonal points that define a "box";
                           ^   |------max
                           |   |       |
                           |   |       |
                           |  min------|
                           |------------->
    epsilon - allowed linear calculation error: if the distance between 2 points
              is less than epsilon, they are equal;
    zero_axis - the axis along which the line normal is expected to have zero
                coordinate.
  The corner case is the case where the line is either horizontal or vertical.
  """
  other_axis = 1 - zero_axis
  theta, theta_0 = equation
  point = np.array([[0.0], [0.0]])
  point[other_axis, 0] =  -theta_0 / theta[other_axis, 0]
  # Broadcasting. Getting 2 points with the correctly set other axes.
  points = np.dot(point, np.array([[1, 1]]))
  # Filling zero axes.
  points[zero_axis, 0] = min_point[zero_axis, 0]
  points[zero_axis, 1] = max_point[zero_axis, 0]
  return points
