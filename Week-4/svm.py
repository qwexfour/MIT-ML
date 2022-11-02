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

import gradient_descent as gd

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

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
  """Returns the derivative of hinge(v).

  Parameters:
    v - a row vector of alternative hinge scalar inputs.
  Returns a row vector of hinge derivatives corresponding to the provided
  values.
  """
  return np.where(v < 1, -1, 0)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
  """Calculates hinge loss gradient for every provided labeled data point
  with respect to theta parameter in th point.

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - hyperplane theta parameter ([d x 1] numpy array of numbers);
    th0 - hyperplane theta_0 parameter ([1 x 1] numpy array of numbers,
          or a single number).
  Returns row vector of gradients for each point, which gives [d x n] matrix.
  """
  kinda_margin = y * hp.substitute(x, th, th0)
  return d_hinge(kinda_margin) * y * x

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
  """Calculates hinge loss partial derivative for every provided labeled data
  point with respect to theta_0 parameter in th0 point.

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - hyperplane theta parameter ([d x 1] numpy array of numbers);
    th0 - hyperplane theta_0 parameter ([1 x 1] numpy array of numbers,
          or a single number).
  Returns row vector of partial derivatives for each point ([1 x n] matrix).
  """
  kinda_margin = y * hp.substitute(x, th, th0)
  return d_hinge(kinda_margin) * y

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
  """Calculates SVM objective function gradient with respect to theta
  parameter in th point.

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - hyperplane theta parameter ([d x 1] numpy array of numbers);
    th0 - hyperplane theta_0 parameter ([1 x 1] numpy array of numbers,
          or a single number);
    lam - regularization parameter, a number.
  Returns the gradient ([d x 1] numpy array).
  """
  avg = np.mean(d_hinge_loss_th(x, y, th, th0), axis=1, keepdims=True)
  return avg + 2 * lam * th

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
  """Calculates SVM objective function partial derivative with respect to
  theta_0 parameter in th0 point.

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - hyperplane theta parameter ([d x 1] numpy array of numbers);
    th0 - hyperplane theta_0 parameter ([1 x 1] numpy array of numbers,
          or a single number);
    lam - regularization parameter, a number.
  Returns the partial derivative ([1 x 1] numpy array).
  """
  return np.mean(d_hinge_loss_th0(x, y, th, th0), axis=1, keepdims=True)

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(x, y, th, th0, lam):
  """Calculates SVM objective function gradient with respect to theta
  and theta_0 parameters in (th, th0) point.

  Parameters:
    x - n data points in d dimensions ([d x n] numpy array of numbers);
    y - data labels ([1 x n] numpy array of elements in {+1, -1}, or a
        single number +1 or -1 for n == 1);
    th - hyperplane theta parameter ([d x 1] numpy array of numbers);
    th0 - hyperplane theta_0 parameter ([1 x 1] numpy array of numbers,
          or a single number);
    lam - regularization parameter, a number.
  Returns the gradient ([d+1 x 1] numpy array).
  """
  return np.vstack((d_svm_obj_th(x, y, th, th0, lam),
                    d_svm_obj_th0(x, y, th, th0, lam)))

def batch_svm_min(data, labels, lam):
  """Minimizes SVM objective for the provided data."""
  def svm_min_step_size_fn(i):
    return 2/(i+1)**0.5
  d = data.shape[0]
  return gd.gd(lambda th_th0 : svm_obj(data, labels, th_th0[0:-1, :],
                                       th_th0[-1:, :], lam),
               lambda th_th0 : svm_obj_grad(data, labels, th_th0[0:-1, :],
                                            th_th0[-1:, :], lam),
               np.zeros((d + 1, 1)),
               svm_min_step_size_fn,
               10)

def _super_simple_separable_svm_obj_test():
  x_1 = np.array([[2, 3, 9, 12],
                [5, 2, 6, 5]])
  y_1 = np.array([[1, -1, 1, -1]])

  th1, th1_0 = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

  # Test case 1
  assert abs(svm_obj(x_1, y_1, th1, th1_0, .1) - 0.1566839689) < 0.0000000001

  # Test case 2
  assert abs(svm_obj(x_1, y_1, th1, th1_0, 0.0)) < 0.0000000001

def _svm_obj_grad_test():
  X1 = np.array([[1, 2, 3, 9, 10]])
  y1 = np.array([[1, 1, 1, -1, -1]])
  th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
  X2 = np.array([[2, 3, 9, 12],
                 [5, 2, 6, 5]])
  y2 = np.array([[1, -1, 1, -1]])
  th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

  assert d_hinge(np.array([[ 71.]])).tolist() == [[0]]
  assert d_hinge(np.array([[ -23.]])).tolist() == [[-1]]
  assert d_hinge(np.array([[ 71, -23.]])).tolist() == [[0, -1]]

  assert d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist() == [[0], [0]]
  assert d_hinge_loss_th(X2, y2, th2, th20).tolist() == \
         [[0, 3, 0, 12], [0, 2, 0, 5]]
  assert d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist() == [[0]]
  assert d_hinge_loss_th0(X2, y2, th2, th20).tolist() == [[0, 1, 0, 1]]

  assert d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist() == \
         [[-0.06], [0.3]]
  assert d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist() == [[3.69], [2.05]]
  assert d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist() == [[0.0]]
  assert d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist() == [[0.5]]

  assert svm_obj_grad(X2, y2, th2, th20, 0.01).tolist() == [[3.69], [2.05], [0.5]]
  assert svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist() == \
         [[-0.06], [0.3], [0.0]]

def _separable_medium_batch_svm_min_test():
  x = np.array([[2, -1, 1, 1],
                [-2, 2, 2, -1]])
  y = np.array([[1, -1, 1, -1]])
  res = batch_svm_min(x, y, 0.0001)
  ref = np.array([[1.44606931], [0.7975608], [-1.20825111]])
  assert np.linalg.norm(res[0] - ref) < 0.00001

def _main():
  _super_simple_separable_svm_obj_test()
  _svm_obj_grad_test()
  _separable_medium_batch_svm_min_test()

if __name__ == "__main__":
  _main()
