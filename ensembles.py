# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Ensembles of place and head direction cells.

These classes provide the targets for the training of grid-cell networks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def one_hot_max(x, axis=-1):
  """Compute one-hot vectors setting to one the index with the maximum value."""
  return tf.one_hot(tf.argmax(x, axis=axis),
                    depth=x.get_shape()[-1],
                    dtype=x.dtype)


def softmax(x, axis=-1):
  """Compute softmax values for each sets of scores in x."""
  return tf.nn.softmax(x, dim=axis)


def softmax_sample(x):
  """Sample the categorical distribution from logits and sample it."""
  dist = tf.contrib.distributions.OneHotCategorical(logits=x, dtype=tf.float32)
  return dist.sample()


class CellEnsemble(object):
  """Abstract parent class for place and head direction cell ensembles."""

  def __init__(self, n_cells, soft_targets, soft_init):
    self.n_cells = n_cells
    if soft_targets not in ["softmax", "voronoi", "sample", "normalized"]:
      raise ValueError
    else:
      self.soft_targets = soft_targets
    # Provide initialization of LSTM in the same way as targets if not specified
    # i.e one-hot if targets are Voronoi
    if soft_init is None:
      self.soft_init = soft_targets
    else:
      if soft_init not in [
          "softmax", "voronoi", "sample", "normalized", "zeros"
      ]:
        raise ValueError
      else:
        self.soft_init = soft_init

  def get_targets(self, x):
    """Type of target."""

    if self.soft_targets == "normalized":
      targets = tf.exp(self.unnor_logpdf(x))
    elif self.soft_targets == "softmax":
      lp = self.log_posterior(x)
      targets = softmax(lp)
    elif self.soft_targets == "sample":
      lp = self.log_posterior(x)
      targets = softmax_sample(lp)
    elif self.soft_targets == "voronoi":
      lp = self.log_posterior(x)
      targets = one_hot_max(lp)
    return targets

  def get_init(self, x):
    """Type of initialisation."""

    if self.soft_init == "normalized":
      init = tf.exp(self.unnor_logpdf(x))
    elif self.soft_init == "softmax":
      lp = self.log_posterior(x)
      init = softmax(lp)
    elif self.soft_init == "sample":
      lp = self.log_posterior(x)
      init = softmax_sample(lp)
    elif self.soft_init == "voronoi":
      lp = self.log_posterior(x)
      init = one_hot_max(lp)
    elif self.soft_init == "zeros":
      init = tf.zeros_like(self.unnor_logpdf(x))
    return init

  def loss(self, predictions, targets):
    """Loss."""

    if self.soft_targets == "normalized":
      smoothing = 1e-2
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=(1. - smoothing) * targets + smoothing * 0.5,
          logits=predictions,
          name="ensemble_loss")
      loss = tf.reduce_mean(loss, axis=-1)
    else:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=targets,
          logits=predictions,
          name="ensemble_loss")
    return loss

  def log_posterior(self, x):
    logp = self.unnor_logpdf(x)
    log_posteriors = logp - tf.reduce_logsumexp(logp, axis=2, keep_dims=True)
    return log_posteriors


class PlaceCellEnsemble(CellEnsemble):
  """Calculates the dist over place cells given an absolute position."""

  def __init__(self, n_cells, stdev=0.35, pos_min=-5, pos_max=5, seed=None,
               soft_targets=None, soft_init=None):
    super(PlaceCellEnsemble, self).__init__(n_cells, soft_targets, soft_init)
    # Create a random MoG with fixed cov over the position (Nx2)
    rs = np.random.RandomState(seed)
    self.means = rs.uniform(pos_min, pos_max, size=(self.n_cells, 2))
    self.variances = np.ones_like(self.means) * stdev**2

  def unnor_logpdf(self, trajs):
    # Output the probability of each component at each point (BxTxN)
    diff = trajs[:, :, tf.newaxis, :] - self.means[np.newaxis, np.newaxis, ...]
    unnor_logp = -0.5 * tf.reduce_sum((diff**2)/ self.variances, axis=-1)
    return unnor_logp


class HeadDirectionCellEnsemble(CellEnsemble):
  """Calculates the dist over HD cells given an absolute angle."""

  def __init__(self, n_cells, concentration=20, seed=None,
               soft_targets=None, soft_init=None):
    super(HeadDirectionCellEnsemble, self).__init__(n_cells,
                                                    soft_targets,
                                                    soft_init)
    # Create a random Von Mises with fixed cov over the position
    rs = np.random.RandomState(seed)
    self.means = rs.uniform(-np.pi, np.pi, (n_cells))
    self.kappa = np.ones_like(self.means) * concentration

  def unnor_logpdf(self, x):
    return self.kappa * tf.cos(x - self.means[np.newaxis, np.newaxis, :])
