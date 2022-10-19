#!/usr/bin/python3
"""A program that implements HW3 tasks.

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
# The code was downloaded from
# <https://introml_oll.odl.mit.edu/cat-soop/_static/6.036/homework/hw03/
#  code_and_data_for_hw3.zip>.
# All the used assets should be downloaded from there too.

import pdb
import argparse
import numpy as np
import code_for_hw3_part2 as hw3

def main():
  args = parse_args()
  if args.auto:
    process_auto()
  if args.review:
    process_review()
  if args.mnist:
    process_mnist()

def parse_args():
  parser = argparse.ArgumentParser(description='I will perform tasks from HW3.')
  parser.add_argument('--auto', action='store_true',
                      help='work on auto data set')
  parser.add_argument('--review', action='store_true',
                      help='work on review data set')
  parser.add_argument('--mnist', action='store_true',
                      help='work on mnist data set')
  return parser.parse_args()

def process_auto():
  #-------------------------------------------------------------------------------
  # Auto Data
  #-------------------------------------------------------------------------------

  # Returns a list of dictionaries.  Keys are the column names, including mpg.
  auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

  # The choice of feature processing for each feature, mpg is always raw and
  # does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
  # 'name' is not numeric and would need a different encoding.
  feature_sets = [[('cylinders', hw3.raw),
                   ('displacement', hw3.raw),
                   ('horsepower', hw3.raw),
                   ('weight', hw3.raw),
                   ('acceleration', hw3.raw),
                   ## Drop model_year by default
                   ## ('model_year', hw3.raw),
                   ('origin', hw3.raw)],
                  [('cylinders', hw3.one_hot),
                   ('displacement', hw3.standard),
                   ('horsepower', hw3.standard),
                   ('weight', hw3.standard),
                   ('acceleration', hw3.standard),
                   ## Drop model_year by default
                   ## ('model_year', hw3.standard),
                   ('origin', hw3.one_hot)]]
  Ts = [1, 10, 50]

  #-------------------------------------------------------------------------------
  # Analyze auto data
  #-------------------------------------------------------------------------------

  # Your code here to process the auto data

  for f_set_idx, features in enumerate(feature_sets):
    for T in Ts:
      # Construct the standard data and label arrays
      auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
      ptron_score = hw3.xval_learning_alg(hw3.perceptron, auto_data,
                                          auto_labels, 10, T)
      av_ptron_score = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data,
                                             auto_labels, 10, T)
      print(f'Analysis for auto data for feature set {f_set_idx+1} and T = {T}:')
      print('  auto data and labels shape', auto_data.shape, auto_labels.shape)
      print('  Perceptron score is ', ptron_score)
      print('  Averaged perceptron score is ', av_ptron_score)

  auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, feature_sets[1])
  auto_theta = hw3.averaged_perceptron(auto_data, auto_labels, {'T' : 1})
  print('The best separator using averaged perceptron, T = 1, the first feature '
        'set:')
  print(*auto_theta)

def process_review():
  #-------------------------------------------------------------------------------
  # Review Data
  #-------------------------------------------------------------------------------

  # Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
  # The train data has 10,000 examples
  review_data = hw3.load_review_data('reviews.tsv')

  # Lists texts of reviews and list of labels (1 or -1)
  review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

  # The dictionary of all the words for "bag of words"
  dictionary = hw3.bag_of_words(review_texts)

  # The standard data arrays for the bag of words
  review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
  review_labels = hw3.rv(review_label_list)
  print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

  #-------------------------------------------------------------------------------
  # Analyze review data
  #-------------------------------------------------------------------------------

  # Your code here to process the review data

def process_mnist():
  #-------------------------------------------------------------------------------
  # MNIST Data
  #-------------------------------------------------------------------------------

  """
  Returns a dictionary formatted as follows:
  {
      0: {
          "images": [(m by n image), (m by n image), ...],
          "labels": [0, 0, ..., 0]
      },
      1: {...},
      ...
      9
  }
  Where labels range from 0 to 9 and (m, n) images are represented
  by arrays of floats from 0 to 1
  """
  mnist_data_all = hw3.load_mnist_data(range(10))

  print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

  raw_features_task = [(0, 1), (2, 4), (6, 8), (9, 0)]

  print('Raw data tests:')
  for fst_digit, snd_digit in raw_features_task:
    print(f'  Comparaing {fst_digit} and {snd_digit}.')
    d0 = mnist_data_all[fst_digit]["images"]
    d1 = mnist_data_all[snd_digit]["images"]
    y0 = np.repeat(-1, len(d0)).reshape(1,-1)
    y1 = np.repeat(1, len(d1)).reshape(1,-1)

    # data goes into the feature computation functions
    data = np.vstack((d0, d1))
    # labels can directly go into the perceptron algorithm
    labels = np.vstack((y0.T, y1.T)).T

    # use this function to evaluate accuracy
    acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
    print('  Accuracy:', acc)

  #-------------------------------------------------------------------------------
  # Analyze MNIST data
  #-------------------------------------------------------------------------------

  # Your code here to process the MNIST data

def raw_mnist_features(x):
  """
  @param x (n_samples,m,n) array with values in (0,1)
  @return (m*n,n_samples) reshaped array where each entry is preserved
  """
  n_samples, m, n = x.shape
  x = x.reshape(n_samples, m * n)
  return x.T

def row_average_features(x):
  """
  This should either use or modify your code from the tutor questions.

  @param x (n_samples,m,n) array with values in (0,1)
  @return (m,n_samples) array where each entry is the average of a row
  """
  raise Exception("modify me!")


def col_average_features(x):
  """
  This should either use or modify your code from the tutor questions.

  @param x (n_samples,m,n) array with values in (0,1)
  @return (n,n_samples) array where each entry is the average of a column
  """
  raise Exception("modify me!")


def top_bottom_features(x):
  """
  This should either use or modify your code from the tutor questions.

  @param x (n_samples,m,n) array with values in (0,1)
  @return (2,n_samples) array where the first entry of each column is the average of the
  top half of the image = rows 0 to floor(m/2) [exclusive]
  and the second entry is the average of the bottom half of the image
  = rows floor(m/2) [inclusive] to m
  """
  raise Exception("modify me!")

if __name__ == "__main__":
  main()
