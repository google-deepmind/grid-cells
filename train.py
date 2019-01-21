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

"""Supervised training for the Grid cell network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import numpy as np
import tensorflow as tf
import Tkinter  # pylint: disable=unused-import

matplotlib.use('Agg')

import dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import model  # pylint: disable=g-bad-import-order
import scores  # pylint: disable=g-bad-import-order
import utils  # pylint: disable=g-bad-import-order


# Task config
tf.flags.DEFINE_string('task_dataset_info', 'square_room',
                       'Name of the room in which the experiment is performed.')
tf.flags.DEFINE_string('task_root',
                       None,
                       'Dataset path.')
tf.flags.DEFINE_float('task_env_size', 2.2,
                      'Environment size (meters).')
tf.flags.DEFINE_list('task_n_pc', [256],
                     'Number of target place cells.')
tf.flags.DEFINE_list('task_pc_scale', [0.01],
                     'Place cell standard deviation parameter (meters).')
tf.flags.DEFINE_list('task_n_hdc', [12],
                     'Number of target head direction cells.')
tf.flags.DEFINE_list('task_hdc_concentration', [20.],
                     'Head direction concentration parameter.')
tf.flags.DEFINE_integer('task_neurons_seed', 8341,
                        'Seeds.')
tf.flags.DEFINE_string('task_targets_type', 'softmax',
                       'Type of target, soft or hard.')
tf.flags.DEFINE_string('task_lstm_init_type', 'softmax',
                       'Type of LSTM initialisation, soft or hard.')
tf.flags.DEFINE_bool('task_velocity_inputs', True,
                     'Input velocity.')
tf.flags.DEFINE_list('task_velocity_noise', [0.0, 0.0, 0.0],
                     'Add noise to velocity.')

# Model config
tf.flags.DEFINE_integer('model_nh_lstm', 128, 'Number of hidden units in LSTM.')
tf.flags.DEFINE_integer('model_nh_bottleneck', 256,
                        'Number of hidden units in linear bottleneck.')
tf.flags.DEFINE_list('model_dropout_rates', [0.5],
                     'List of floats with dropout rates.')
tf.flags.DEFINE_float('model_weight_decay', 1e-5,
                      'Weight decay regularisation')
tf.flags.DEFINE_bool('model_bottleneck_has_bias', False,
                     'Whether to include a bias in linear bottleneck')
tf.flags.DEFINE_float('model_init_weight_disp', 0.0,
                      'Initial weight displacement.')

# Training config
tf.flags.DEFINE_integer('training_epochs', 1000, 'Number of training epochs.')
tf.flags.DEFINE_integer('training_steps_per_epoch', 1000,
                        'Number of optimization steps per epoch.')
tf.flags.DEFINE_integer('training_minibatch_size', 10,
                        'Size of the training minibatch.')
tf.flags.DEFINE_integer('training_evaluation_minibatch_size', 4000,
                        'Size of the minibatch during evaluation.')
tf.flags.DEFINE_string('training_clipping_function', 'utils.clip_all_gradients',
                       'Function for gradient clipping.')
tf.flags.DEFINE_float('training_clipping', 1e-5,
                      'The absolute value to clip by.')

tf.flags.DEFINE_string('training_optimizer_class', 'tf.train.RMSPropOptimizer',
                       'The optimizer used for training.')
tf.flags.DEFINE_string('training_optimizer_options',
                       '{"learning_rate": 1e-5, "momentum": 0.9}',
                       'Defines a dict with opts passed to the optimizer.')

# Store
tf.flags.DEFINE_string('saver_results_directory',
                       None,
                       'Path to directory for saving results.')
tf.flags.DEFINE_integer('saver_eval_time', 2,
                        'Frequency at which results are saved.')

# Require flags
tf.flags.mark_flag_as_required('task_root')
tf.flags.mark_flag_as_required('saver_results_directory')
FLAGS = tf.flags.FLAGS


def train():
  """Training loop."""

  tf.reset_default_graph()

  # Create the motion models for training and evaluation
  data_reader = dataset_reader.DataReader(
      FLAGS.task_dataset_info, root=FLAGS.task_root, num_threads=4)
  train_traj = data_reader.read(batch_size=FLAGS.training_minibatch_size)

  # Create the ensembles that provide targets during training
  place_cell_ensembles = utils.get_place_cell_ensembles(
      env_size=FLAGS.task_env_size,
      neurons_seed=FLAGS.task_neurons_seed,
      targets_type=FLAGS.task_targets_type,
      lstm_init_type=FLAGS.task_lstm_init_type,
      n_pc=FLAGS.task_n_pc,
      pc_scale=FLAGS.task_pc_scale)

  head_direction_ensembles = utils.get_head_direction_ensembles(
      neurons_seed=FLAGS.task_neurons_seed,
      targets_type=FLAGS.task_targets_type,
      lstm_init_type=FLAGS.task_lstm_init_type,
      n_hdc=FLAGS.task_n_hdc,
      hdc_concentration=FLAGS.task_hdc_concentration)
  target_ensembles = place_cell_ensembles + head_direction_ensembles

  # Model creation
  rnn_core = model.GridCellsRNNCell(
      target_ensembles=target_ensembles,
      nh_lstm=FLAGS.model_nh_lstm,
      nh_bottleneck=FLAGS.model_nh_bottleneck,
      dropoutrates_bottleneck=np.array(FLAGS.model_dropout_rates),
      bottleneck_weight_decay=FLAGS.model_weight_decay,
      bottleneck_has_bias=FLAGS.model_bottleneck_has_bias,
      init_weight_disp=FLAGS.model_init_weight_disp)
  rnn = model.GridCellsRNN(rnn_core, FLAGS.model_nh_lstm)

  # Get a trajectory batch
  input_tensors = []
  init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
  if FLAGS.task_velocity_inputs:
    # Add the required amount of noise to the velocities
    vel_noise = tf.distributions.Normal(0.0, 1.0).sample(
        sample_shape=ego_vel.get_shape()) * FLAGS.task_velocity_noise
    input_tensors = [ego_vel + vel_noise] + input_tensors
  # Concatenate all inputs
  inputs = tf.concat(input_tensors, axis=2)

  # Replace euclidean positions and angles by encoding of place and hd ensembles
  # Note that the initial_conds will be zeros if the ensembles were configured
  # to provide that type of initialization
  initial_conds = utils.encode_initial_conditions(
      init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)

  # Encode targets as well
  ensembles_targets = utils.encode_targets(
      target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

  # Estimate future encoding of place and hd ensembles inputing egocentric vels
  outputs, _ = rnn(initial_conds, inputs, training=True)
  ensembles_logits, bottleneck, lstm_output = outputs

  # Training loss
  pc_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=ensembles_targets[0], logits=ensembles_logits[0], name='pc_loss')
  hd_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=ensembles_targets[1], logits=ensembles_logits[1], name='hd_loss')
  total_loss = pc_loss + hd_loss
  train_loss = tf.reduce_mean(total_loss, name='train_loss')

  # Optimisation ops
  optimizer_class = eval(FLAGS.training_optimizer_class)  # pylint: disable=eval-used
  optimizer = optimizer_class(**eval(FLAGS.training_optimizer_options))  # pylint: disable=eval-used
  grad = optimizer.compute_gradients(train_loss)
  clip_gradient = eval(FLAGS.training_clipping_function)  # pylint: disable=eval-used
  clipped_grad = [
      clip_gradient(g, var, FLAGS.training_clipping) for g, var in grad
  ]
  train_op = optimizer.apply_gradients(clipped_grad)

  # Store the grid scores
  grid_scores = dict()
  grid_scores['btln_60'] = np.zeros((FLAGS.model_nh_bottleneck,))
  grid_scores['btln_90'] = np.zeros((FLAGS.model_nh_bottleneck,))
  grid_scores['btln_60_separation'] = np.zeros((FLAGS.model_nh_bottleneck,))
  grid_scores['btln_90_separation'] = np.zeros((FLAGS.model_nh_bottleneck,))
  grid_scores['lstm_60'] = np.zeros((FLAGS.model_nh_lstm,))
  grid_scores['lstm_90'] = np.zeros((FLAGS.model_nh_lstm,))

  # Create scorer objects
  starts = [0.2] * 10
  ends = np.linspace(0.4, 1.0, num=10)
  masks_parameters = zip(starts, ends.tolist())
  latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(),
                                          masks_parameters)

  with tf.train.SingularMonitoredSession() as sess:
    for epoch in range(FLAGS.training_epochs):
      loss_acc = list()
      for _ in range(FLAGS.training_steps_per_epoch):
        res = sess.run({'train_op': train_op, 'total_loss': train_loss})
        loss_acc.append(res['total_loss'])

      tf.logging.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
                      np.mean(loss_acc), np.std(loss_acc))
      if epoch % FLAGS.saver_eval_time == 0:
        res = dict()
        for _ in xrange(FLAGS.training_evaluation_minibatch_size //
                        FLAGS.training_minibatch_size):
          mb_res = sess.run({
              'bottleneck': bottleneck,
              'lstm': lstm_output,
              'pos_xy': target_pos
          })
          res = utils.concat_dict(res, mb_res)

        # Store at the end of validation
        filename = 'rates_and_sac_latest_hd.pdf'
        grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
            'btln_60_separation'], grid_scores[
                'btln_90_separation'] = utils.get_scores_and_plot(
                    latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                    FLAGS.saver_results_directory, filename)


def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train()

if __name__ == '__main__':
  tf.app.run()
