#!/usr/bin/python3
"""Program to calculate results for the 5th week exercises.

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

import square_loss as sqr

def _task_2_data():
  blue = (np.array([[1]]), 0)
  green = (np.array([[1]]), 1)
  x = np.array([[1, 1, 3, 3]])
  y = np.array([[3, 1, 2, 6]])
  return blue, green, x, y

def _task_2_1():
  blue, green, x, y = _task_2_data()
  res = sqr.square_loss(x, y, *blue)
  print("Ex2.1:")
  print(res)

def _task_2_2():
  blue, green, x, y = _task_2_data()
  res = sqr.square_loss_grad(x, y, *blue)
  print("Ex2.2(gradients are column vectors, dL/dth0 is the last element):")
  print(res)

def _task_2_3():
  blue, green, x, y = _task_2_data()
  res = sqr.square_loss(x, y, *green)
  print("Ex2.3:")
  print(res)

def _task_2_4():
  blue, green, x, y = _task_2_data()
  res = sqr.square_loss_grad(x, y, *green)
  print("Ex2.4(gradients are column vectors, dL/dth0 is the last element):")
  print(res)

def _hw_4():
  X = np.array([[1, 2], [2, 3], [3, 5], [1, 4]])
  X_sqr = np.dot(X, np.transpose(X))
  det = np.linalg.det(X_sqr)
  print("HW4: det(X@X.T) = ", 0)

def _main():
  _task_2_1()
  _task_2_2()
  _task_2_3()
  _task_2_4()
  _hw_4()

if __name__ == "__main__":
  _main()
