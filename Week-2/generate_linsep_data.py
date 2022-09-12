#!/usr/bin/python3
"""A simple program for generating linearly separable data.

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

import linseptools as lst
import argparse

def main():
  args = parse_args()
  points, labels, theta, theta_0 = lst.generate_input(2, args.num_points)
  lst.draw(points, labels, (theta, theta_0))

def parse_args():
  parser = argparse.ArgumentParser(description='I will generate linearly '
                                               'separable data for you.')
  parser.add_argument('num_points', metavar='N', type=int, default=100,
                      nargs='?', help='The number of points to generate')
  return parser.parse_args()

if __name__ == "__main__":
  main()
