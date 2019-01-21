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

"""Model for grid cells supervised training.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import sonnet as snt
import tensorflow as tf


def displaced_linear_initializer(input_size, displace, dtype=tf.float32):
  stddev = 1. / numpy.sqrt(input_size)
  return tf.truncated_normal_initializer(
      mean=displace*stddev, stddev=stddev, dtype=dtype)


class GridCellsRNNCell(snt.RNNCore):
  """LSTM core implementation for the grid cell network."""

  def __init__(self,
               target_ensembles,
               nh_lstm,
               nh_bottleneck,
               nh_embed=None,
               dropoutrates_bottleneck=None,
               bottleneck_weight_decay=0.0,
               bottleneck_has_bias=False,
               init_weight_disp=0.0,
               name="grid_cells_core"):
    """Constructor of the RNN cell.

    Args:
      target_ensembles: Targets, place cells and head direction cells.
      nh_lstm: Size of LSTM cell.
      nh_bottleneck: Size of the linear layer between LSTM output and output.
      nh_embed: Number of hiddens between input and LSTM input.
      dropoutrates_bottleneck: Iterable of keep rates (0,1]. The linear layer is
        partitioned into as many groups as the len of this parameter.
      bottleneck_weight_decay: Weight decay used in the bottleneck layer.
      bottleneck_has_bias: If the bottleneck has a bias.
      init_weight_disp: Displacement in the weights initialisation.
      name: the name of the module.
    """
    super(GridCellsRNNCell, self).__init__(name=name)
    self._target_ensembles = target_ensembles
    self._nh_embed = nh_embed
    self._nh_lstm = nh_lstm
    self._nh_bottleneck = nh_bottleneck
    self._dropoutrates_bottleneck = dropoutrates_bottleneck
    self._bottleneck_weight_decay = bottleneck_weight_decay
    self._bottleneck_has_bias = bottleneck_has_bias
    self._init_weight_disp = init_weight_disp
    self.training = False
    with self._enter_variable_scope():
      self._lstm = snt.LSTM(self._nh_lstm)

  def _build(self, inputs, prev_state):
    """Build the module.

    Args:
      inputs: Egocentric velocity (BxN)
      prev_state: Previous state of the recurrent network

    Returns:
      ((predictions, bottleneck, lstm_outputs), next_state)
      The predictions
    """
    conc_inputs = tf.concat(inputs, axis=1, name="conc_inputs")
    # Embedding layer
    lstm_inputs = conc_inputs
    # LSTM
    lstm_output, next_state = self._lstm(lstm_inputs, prev_state)
    # Bottleneck
    bottleneck = snt.Linear(self._nh_bottleneck,
                            use_bias=self._bottleneck_has_bias,
                            regularizers={
                                "w": tf.contrib.layers.l2_regularizer(
                                    self._bottleneck_weight_decay)},
                            name="bottleneck")(lstm_output)
    if self.training and self._dropoutrates_bottleneck is not None:
      tf.logging.info("Adding dropout layers")
      n_scales = len(self._dropoutrates_bottleneck)
      scale_pops = tf.split(bottleneck, n_scales, axis=1)
      dropped_pops = [tf.nn.dropout(pop, rate, name="dropout")
                      for rate, pop in zip(self._dropoutrates_bottleneck,
                                           scale_pops)]
      bottleneck = tf.concat(dropped_pops, axis=1)
    # Outputs
    ens_outputs = [snt.Linear(
        ens.n_cells,
        regularizers={
            "w": tf.contrib.layers.l2_regularizer(
                self._bottleneck_weight_decay)},
        initializers={
            "w": displaced_linear_initializer(self._nh_bottleneck,
                                              self._init_weight_disp,
                                              dtype=tf.float32)},
        name="pc_logits")(bottleneck)
                   for ens in self._target_ensembles]
    return (ens_outputs, bottleneck, lstm_output), tuple(list(next_state))

  @property
  def state_size(self):
    """Returns a description of the state size, without batch dimension."""
    return self._lstm.state_size

  @property
  def output_size(self):
    """Returns a description of the output size, without batch dimension."""
    return tuple([ens.n_cells for ens in self._target_ensembles] +
                 [self._nh_bottleneck, self._nh_lstm])


class GridCellsRNN(snt.AbstractModule):
  """RNN computes place and head-direction cell predictions from velocities."""

  def __init__(self, rnn_cell, nh_lstm, name="grid_cell_supervised"):
    super(GridCellsRNN, self).__init__(name=name)
    self._core = rnn_cell
    self._nh_lstm = nh_lstm

  def _build(self, init_conds, vels, training=False):
    """Outputs place, and head direction cell predictions from velocity inputs.

    Args:
      init_conds: Initial conditions given by ensemble activatons, list [BxN_i]
      vels:  Translational and angular velocities [BxTxV]
      training: Activates and deactivates dropout

    Returns:
      [logits_i]:
        logits_i: Logits predicting i-th ensemble activations (BxTxN_i)
    """
    # Calculate initialization for LSTM. Concatenate pc and hdc activations
    concat_init = tf.concat(init_conds, axis=1)

    init_lstm_state = snt.Linear(self._nh_lstm, name="state_init")(concat_init)
    init_lstm_cell = snt.Linear(self._nh_lstm, name="cell_init")(concat_init)
    self._core.training = training

    # Run LSTM
    output_seq, final_state = tf.nn.dynamic_rnn(cell=self._core,
                                                inputs=(vels,),
                                                time_major=False,
                                                initial_state=(init_lstm_state,
                                                               init_lstm_cell))
    ens_targets = output_seq[:-2]
    bottleneck = output_seq[-2]
    lstm_output = output_seq[-1]
    # Return
    return (ens_targets, bottleneck, lstm_output), final_state

  def get_all_variables(self):
    return (super(GridCellsRNN, self).get_variables()
            + self._core.get_variables())
