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

def main():
  args = parse_args()
  run_test(args)

def parse_args():
  parser = argparse.ArgumentParser(description='A program that tests '
                                               'perceptron algorithm.')
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
               silent - whether to disable the printouts.
  """
  points, labels, _, _ = lst.generate_input(params.dims, params.points)
  if not params.silent:
    print('{n} {d}D points were generated.'.format(n=params.points,
                                                   d=params.dims))
  theta, theta_0 = petron.perceptron(points, labels, {'T': params.petron_iter})
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

if __name__ == "__main__":
  main()
