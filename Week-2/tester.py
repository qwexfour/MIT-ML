#!/usr/bin/python3
"""A program that tests perceptron algorithm.

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

import argparse
import sys

# FIXME: find a better way.
sys.path.append('../Week-1')
import linseptools as lst
import perceptron as petron
import hyperplane as hpl
import numpy as np

def main():
  args = parse_args()
  run_test(args)

class AlgoAction(argparse.Action):
  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    super().__init__(option_strings, dest, **kwargs)

  def __call__(self, parser, namespace, values, option_string=None):
    if values == 'petron':
      setattr(namespace, 'algo', petron.perceptron)
    elif values == 'av_petron':
      setattr(namespace, 'algo', petron.averaged_perceptron)
    else:
      assert 0, 'An argument choice handler was forgotten'

def parse_args():
  parser = argparse.ArgumentParser(description='A program that tests '
                                               'perceptron algorithm.')
  parser.add_argument('--algo', '-a', choices=['petron', 'av_petron'],
                      default=petron.perceptron, action=AlgoAction,
                      help='which algorithm to call (default: petron)')
  parser.add_argument('--points', '-p', metavar='P', type=int, default=100,
                      help='the number of points to generate')
  parser.add_argument('--dims', '-d', metavar='D', type=int, default=2,
                      help='the number of dimensions')
  parser.add_argument('--petron-iter', '-t', metavar='T', type=int, default=100,
                      help='the number of iterations in perceptron algorithm')
  parser.add_argument('--visualize', '-v', action='store_true',
                      help='use visualization mode (works only for 2D)')
  parser.add_argument('--silent', '-s', action='store_true',
                      help='disable all printouts')
  parser.add_argument('--dump-input', metavar='filename.txt', default='',
                      help='the name of a file to which the generated points '
                           'and labels must be dumped')
  parser.add_argument('--read-input', metavar='filename.txt', default='',
                      help='the name of a file with the input points and '
                           'their labels, the number of points and dimensions '
                           'are derived from the data in this case')
  args = parser.parse_args()

  # Checking restrictions.
  if args.points < 1:
    sys.exit('The number of points must be positive.')
  if args.dims < 1:
    sys.exit('The number of dimensions must be positive.')
  if args.petron_iter < 1:
    sys.exit('The number of iterations in perceptron algorithm must be '
             'positive.')
  if args.visualize and args.dims != 2:
    sys.exit('Visualize mode supported only for 2D task.')
  return args

def run_test(params):
  """Tests perceptron algorithm.

  This function generates some data for perceptron algorithm, runs it and
  scores the result.

  Parameters:
    params - a class with the required parameters:
               points - the number of points to generate,
               dims - the number of dimensions,
               petron_iter - the number of iterations in perceptron algorithm,
               visualize - whether to visualize the result,
               dump_input - a name of a file to which the input data should be
                            dumped,
               read_input - a name of a file with the input data, the data will
                            be read in this case (not generated),
               silent - whether to disable the printouts.
  """
  points, labels = _get_input(params)
  if not params.silent:
    print('{n} {d}D points will be used.'.format(n=params.points,
                                                 d=params.dims))
  if params.dump_input:
    np.savetxt(params.dump_input,
               np.concatenate((points, labels), casting='no'))

  theta, theta_0 = params.algo(points, labels, {'T': params.petron_iter})
  if not params.silent:
    print('{a} was used.'.format(a=params.algo))
  if not params.silent:
    print("Perceptron's solution (for T={t}) is:".format(t=params.petron_iter))
    print('  theta=\n{th},'.format(th=theta))
    print('  theta_0={th0}.'.format(th0=theta_0))
  score = hpl.score(points, labels, theta, theta_0)[0]
  if not params.silent:
    print('{s} out of {p} points were correctly '
          'classified.'.format(s=score, p=params.points))
  if params.visualize and params.dims == 2:
    lst.draw(points, labels, (theta, theta_0))

def _get_input(params):
  """Reads from a file or generates input data.

  Parameters:
    params - a class with the required parameters:
               points - the number of points to generate,
               dims - the number of dimensions,
               read_input - a name of a file with the input data, the data will
                            be read in this case (not generated),
               silent - whether to disable the printouts.
  Reads input data from a file when the read_input parameter is provided,
  otherwise generates it according to the dims and points parameters. The dims
  and points attributes of the provided params are changed according to the data
  when it is read from a file.
  Result:
    Points (numpy [num_dims x num_points] array);
    Labels for those points (numpy [1 x num_points] array);
  """
  if params.read_input:
    data = np.loadtxt(params.read_input)
    points = data[0:-1, :]
    labels = data[-1:, :]
    params.dims = points.shape[0]
    params.points = points.shape[1]

    if not params.silent:
      print('Input data is read from the dump file.')
    return (points, labels)

  points, labels, _, _ = lst.generate_input(params.dims, params.points)
  if not params.silent:
    print('Input data is generated.')
  return (points, labels)

if __name__ == "__main__":
  main()
