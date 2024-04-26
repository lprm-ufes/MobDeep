from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO: salvar pasta saved_args dentro da pasta experiments/exp_name/test_name/models

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#TODO change directories location
"""
  - evaluations
  - generated_data
  - models
  - plots
  - saved_args
  - summaries
  - synthdata

"""


"""

The hyperparameters used in the model:
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- timesteps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- epochs_before_decay - the number of epochs trained with the initial learning rate
- max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "epochs_before_decay"
- batch_size - the batch size

The hyperparameters that could be used in the model:
- init_scale - the initial scale of the weights

To run:

$ python rnn_gan.py --model small|medium|large --datadir simple-examples/data/ --traindir dir-for-checkpoints-and-plots --select_validation_percentage 0-40 --select_test_percentage 0-40"""

from sklearn.preprocessing import MinMaxScaler

import time, datetime, os, sys
import pickle as pkl
from subprocess import call, Popen

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from glob import glob
# import plotting as p
import gc

# import music_data_utils
# from midi_statistics import get_all_stats
import data_loader

# should hide the tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
flags = tf.flags
logging = tf.logging

flags.DEFINE_string("datadir", None, "Directory to save and load midi music files.")
flags.DEFINE_string("traindir", None, "Directory to save checkpoints and gnuplot files.")
flags.DEFINE_string("filename",None,"File name to train the GAN.")
flags.DEFINE_integer("epochs_per_checkpoint", 1,
                     "How many training epochs to do per checkpoint.")
flags.DEFINE_boolean("log_device_placement", False,           #
                   "Outputs info on device placement.")
flags.DEFINE_string("call_after", None, "Call this command after exit.")
flags.DEFINE_integer("exit_after", 1440,
                     "exit after this many minutes")
flags.DEFINE_integer("select_validation_percentage", None,
                     "Select random percentage of data as validation set.")
flags.DEFINE_integer("select_test_percentage", None,
                     "Select random percentage of data as test set.")               
flags.DEFINE_integer("n_samples",None,"Number of samples in the dataset.")                     
flags.DEFINE_integer("n_features",None,"Number of features in the dataset.")                     
flags.DEFINE_boolean("sample", False,
                     "Sample output from the model. Assume training was already done. Save sample output to file.")
flags.DEFINE_integer("n_synth_sample",None,"Number of samples to synth.")
flags.DEFINE_boolean("scale", False, "Scale the dataset using the MinMaxScaler")                    
flags.DEFINE_integer("works_per_composer", None,
                     "Limit number of works per composer that is loaded.")
flags.DEFINE_boolean("disable_feed_previous", False,
                     "Feed output from previous cell to the input of the next. In the generator.")
flags.DEFINE_float("init_scale", 0.05,                # .1, .04
                   "the initial scale of the weights")
flags.DEFINE_float("learning_rate", 0.1,              # .05,.1,.9 
                   "Learning rate")
flags.DEFINE_float("d_lr_factor", 0.5,                # .5
                   "Learning rate decay")
flags.DEFINE_float("max_grad_norm", 5.0,              # 5.0, 10.0
                   "the maximum permissible norm of the gradient")
flags.DEFINE_float("keep_prob", 0.5,                  # 1.0, .35
                   "Keep probability. 1.0 disables dropout.")
flags.DEFINE_float("lr_decay", 1.0,                   # 1.0
                   "Learning rate decay after each epoch after epochs_before_decay")
flags.DEFINE_integer("num_layers_g", 2,                 # 2
                   "Number of stacked recurrent cells in G.")
flags.DEFINE_integer("num_layers_d", 2,                 # 2
                   "Number of stacked recurrent cells in D.")
flags.DEFINE_integer("timesteps", None,               # 200, 500
                   "Limit song inputs to this number of events.")
flags.DEFINE_integer("meta_layer_size", 200,          # 300, 600
                   "Size of hidden layer for meta information module.")
flags.DEFINE_integer("hidden_size_g", 100,              # 200, 1500
                   "Hidden size for recurrent part of G.")
flags.DEFINE_integer("hidden_size_d", 100,              # 200, 1500
                   "Hidden size for recurrent part of D. Default: same as for G.")
flags.DEFINE_integer("epochs_before_decay", 60,       # 40, 140
                   "Number of epochs before starting to decay.")
flags.DEFINE_integer("max_epoch", 100,                # 500, 500
                   "Number of epochs before stopping training.")
flags.DEFINE_integer("batch_size", 64,                # 10, 20
                   "Batch size.")
flags.DEFINE_integer("biscale_slow_layer_ticks", 8,   # 8
                   "Biscale slow layer ticks. Not implemented yet.")
flags.DEFINE_boolean("multiscale", False,             #
                   "Multiscale RNN. Not implemented.")
flags.DEFINE_integer("pretraining_epochs", 6,        # 20, 40
                   "Number of epochs to run lang-model style pretraining.")
flags.DEFINE_boolean("pretraining_d", False,          #
                   "Train D during pretraining.")
flags.DEFINE_boolean("initialize_d", False,           #
                   "Initialize variables for D, no matter if there are trained versions in checkpoint.")
flags.DEFINE_boolean("ignore_saved_args", False,      #
                   "Tells the program to ignore saved arguments, and instead use the ones provided as CLI arguments.")
flags.DEFINE_boolean("pace_events", False,            #
                   "When parsing input data, insert one dummy event at each quarter note if there is no tone.")
flags.DEFINE_boolean("minibatch_d", False,            #
                   "Adding kernel features for minibatch diversity.")
flags.DEFINE_boolean("unidirectional_d", True,        #
                   "Unidirectional RNN instead of bidirectional RNN for D.")
flags.DEFINE_boolean("profiling", False,              #
                   "Profiling. Writing a timeline.json file in plots dir.")
flags.DEFINE_boolean("float16", False,                #
                   "Use floa16 data type. Otherwise, use float32.")
flags.DEFINE_boolean("check_on_valid",False,
"Check the model on the 'validation' dataset. If the validatin dataset is too large"+
" this could take a considerable amount of time.")                   

flags.DEFINE_boolean("adam", False,                   #
                   "Use Adam optimizer.")
flags.DEFINE_boolean("feature_matching", False,       #
                   "Feature matching objective for G.")
flags.DEFINE_boolean("disable_l2_regularizer", False,       #
                   "L2 regularization on weights.")
flags.DEFINE_float("reg_scale", 1.0,       #
                   "L2 regularization scale.")
flags.DEFINE_boolean("synthetic_chords", False,       #
                   "Train on synthetically generated chords (three tones per event).")
flags.DEFINE_integer("tones_per_cell", 1,             # 2,3
                   "Maximum number of tones to output per RNN cell.")
flags.DEFINE_string("composer", None, "Specify exactly one composer, and train model only on this.")
flags.DEFINE_boolean("generate_meta", False, "Generate the composer and genre as part of output.")
flags.DEFINE_float("random_input_scale", 1.0,       #
                   "Scale of random inputs (1.0=same size as generated features).")
flags.DEFINE_boolean("end_classification", False, "Classify only in ends of D. Otherwise, does classification at every timestep and mean reduce.")
flags.DEFINE_string("testname",None,"Test name for use when plotting")
FLAGS = flags.FLAGS

model_layout_flags = ['filename','n_samples','n_features','timesteps', 'num_layers_g', 'num_layers_d', 'hidden_size_g', 'hidden_size_d', 'disable_feed_previous', 'minibatch_d', 'unidirectional_d', 'feature_matching']
# TODO: FIX function parameters
def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=True,
                  reuse=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
    dropout_keep_prob: The float probability to keep the output of any given sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple, reuse=reuse)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple, reuse=reuse)

  return cell
def restore_flags(save_if_none_found=True):
  if FLAGS.traindir:
    saved_args_dir = os.path.join(FLAGS.traindir, 'models/saved_args')
    if save_if_none_found:
      try: os.makedirs(saved_args_dir)
      except: pass
    for arg in FLAGS.__flags:
      if arg not in model_layout_flags:
        continue
      if FLAGS.ignore_saved_args and os.path.exists(os.path.join(saved_args_dir, arg+'.pkl')):
        print('{:%Y-%m-%d %H:%M:%S}: saved_args: Found {} setting from saved state, but using CLI args ({}) and saving (--ignore_saved_args).'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))
      elif os.path.exists(os.path.join(saved_args_dir, arg+'.pkl')):
        with open(os.path.join(saved_args_dir, arg+'.pkl'), 'rb') as f:
          setattr(FLAGS, arg, pkl.load(f))
          print('{:%Y-%m-%d %H:%M:%S}: saved_args: {} from saved state ({}), ignoring CLI args.'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))
      elif save_if_none_found:
        print('{:%Y-%m-%d %H:%M:%S}: saved_args: Found no {} setting from saved state, using CLI args ({}) and saving.'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))
        with open(os.path.join(saved_args_dir, arg+'.pkl'), 'wb') as f:
            print(getattr(FLAGS, arg),arg)
            pkl.dump(getattr(FLAGS, arg), f)
      else:
        print('{:%Y-%m-%d %H:%M:%S}: saved_args: Found no {} setting from saved state, using CLI args ({}) but not saving.'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg))) 

def data_type():
  return tf.float16 if FLAGS.float16 else tf.float32
  #return tf.float16

def my_reduce_mean(what_to_take_mean_over):
  return tf.reshape(what_to_take_mean_over, shape=[-1])[0]

def linear(inp, output_dim, scope=None, stddev=1.0, reuse_scope=False):
  norm = tf.random_normal_initializer(stddev=stddev)
  const = tf.constant_initializer(0.0)
  with tf.compat.v1.variable_scope(scope or 'linear') as scope:
    scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
    if reuse_scope:
      scope.reuse_variables()
    
    w = tf.compat.v1.get_variable('w', [inp.get_shape()[1], output_dim], initializer=norm, dtype=data_type())
    b = tf.compat.v1.get_variable('b', [output_dim], initializer=const, dtype=data_type())
  return tf.matmul(inp, w) + b

def minibatch(inp, num_kernels=25, kernel_dim=10, scope=None, msg='', reuse_scope=False):
  """
   Borrowed from http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
  """
  with tf.compat.v1.variable_scope(scope or 'minibatch_d') as scope:
    scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
    if reuse_scope:
      scope.reuse_variables()
  
    # inp = tf.Print(inp, [inp],
    #         '{} inp = '.format(msg), summarize=20, first_n=20)
    
    x = tf.sigmoid(linear(inp, num_kernels * kernel_dim, scope))
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    # activation = tf.Print(activation, [activation],
    #         '{} activation = '.format(msg), summarize=20, first_n=20)
    diffs = tf.expand_dims(activation, 3) - \
                tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    # diffs = tf.Print(diffs, [diffs],
    #         '{} diffs = '.format(msg), summarize=20, first_n=20)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    # abs_diffs = tf.Print(abs_diffs, [abs_diffs],
    #         '{} abs_diffs = '.format(msg), summarize=20, first_n=20)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    # minibatch_features = tf.Print(minibatch_features, [tf.reduce_min(minibatch_features), tf.reduce_max(minibatch_features)],
    #         '{} minibatch_features (min,max) = '.format(msg), summarize=20, first_n=20)
  return tf.concat( [inp, minibatch_features],1)

class RNNGAN(object):
  """The RNNGAN model."""

  def __init__(self, is_training, num_song_features=None, num_meta_features=None, session=None):
    batch_size = FLAGS.batch_size
    self.batch_size =  batch_size	
    timesteps = FLAGS.timesteps
    self.timesteps = timesteps
    print('timesteps: {}'.format(self.timesteps))
    self._input_sample_data = tf.compat.v1.placeholder(shape=[batch_size, timesteps, num_song_features], dtype=data_type())
    self._input_metadata = tf.compat.v1.placeholder(shape=[batch_size, num_meta_features], dtype=data_type())
    
    print("self._input_sample_data ",self._input_sample_data, ' timesteps ',timesteps)
    
    sample_data_inputs = [tf.squeeze(input_, [1])
            for input_ in tf.split(self._input_sample_data,timesteps,1)]

    with tf.compat.v1.variable_scope('G') as scope:
      scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
      if is_training and FLAGS.keep_prob < 1:
        cell = make_rnn_cell([FLAGS.hidden_size_g]*FLAGS.num_layers_g,dropout_keep_prob=FLAGS.keep_prob)
      else:
         cell = make_rnn_cell([FLAGS.hidden_size_g]*FLAGS.num_layers_g)	  

      # random_inputs = tf.random.uniform(shape=[batch_size, timesteps, int(num_song_features)], minval=0.0, maxval=1.0, dtype=data_type())
      # self._initial_state = cell.get_initial_state(random_inputs,batch_size)
      self._initial_state = cell.zero_state(batch_size, data_type())

      # TODO: (possibly temporarily) disabling meta info
      if FLAGS.generate_meta:
        metainputs = tf.random.uniform(shape=[batch_size, int(FLAGS.random_input_scale*num_meta_features)], minval=0.0, maxval=1.0)
        meta_g = tf.nn.relu(linear(metainputs, FLAGS.meta_layer_size, scope='meta_layer', reuse_scope=False))
        meta_softmax_w = tf.get_variable("meta_softmax_w", [FLAGS.meta_layer_size, num_meta_features])
        meta_softmax_b = tf.get_variable("meta_softmax_b", [num_meta_features])
        meta_logits = tf.nn.xw_plus_b(meta_g, meta_softmax_w, meta_softmax_b)
        meta_probs = tf.nn.softmax(meta_logits)

      random_rnninputs = tf.random.uniform(shape=[batch_size, timesteps, int(FLAGS.random_input_scale*num_song_features)], minval=0.0, maxval=1.0, dtype=data_type())

      # Make list of tensors. One per step in recurrence.
      # Each tensor is batchsize*numfeatures.
      
      random_rnninputs = [tf.squeeze(input_, [1]) for input_ in tf.split( random_rnninputs,timesteps,1)]
      
      # REAL GENERATOR:
      state = self._initial_state
      # as we feed the output as the input to the next, we 'invent' the initial 'output'.
      generated_point = tf.random.uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0, dtype=data_type())
      outputs = []
      self._generated_features = []
      for i,input_ in enumerate(random_rnninputs):
        if i > 0: scope.reuse_variables()
        concat_values = [input_]
        if not FLAGS.disable_feed_previous:
          concat_values.append(generated_point)
        if FLAGS.generate_meta:
          concat_values.append(meta_probs)
        if len(concat_values):
          input_ = tf.concat(axis=1, values=concat_values)
        input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_g,
                            scope='input_layer', reuse_scope=(i!=0)))
        output, state = cell(input_, state)
        outputs.append(output)
        #generated_point = tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0)))
        generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))
        self._generated_features.append(generated_point)
      

      # PRETRAINING GENERATOR, will feed inputs, not generated outputs:
      scope.reuse_variables()
      # as we feed the output as the input to the next, we 'invent' the initial 'output'.
      prev_target = tf.random.uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0, dtype=data_type())
      outputs = []
      self._generated_features_pretraining = []
      for i,input_ in enumerate(random_rnninputs):
        concat_values = [input_]
        if not FLAGS.disable_feed_previous:
          concat_values.append(prev_target)
        if FLAGS.generate_meta:
          concat_values.append(self._input_metadata)
        if len(concat_values):
          input_ = tf.concat(axis=1, values=concat_values)
        input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_g, scope='input_layer', reuse_scope=(i!=0)))
        output, state = cell(input_, state)
        outputs.append(output)
        #generated_point = tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0)))
        generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))
        self._generated_features_pretraining.append(generated_point)
        prev_target = sample_data_inputs[i]

    self._final_state = state

    # These are used both for pretraining and for D/G training further down.
    self._lr = tf.Variable(FLAGS.learning_rate, trainable=False, dtype=data_type())
    self.g_params = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('model/G/')]
    if FLAGS.adam:
      g_optimizer = tf.train.AdamOptimizer(self._lr)
    else:
      g_optimizer = tf.compat.v1.train.GradientDescentOptimizer(self._lr)
   
    reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.1  # Choose an appropriate one.
    reg_loss = reg_constant * sum(reg_losses)
    
    # ---BEGIN, PRETRAINING. ---
    
    print(tf.transpose(tf.stack(self._generated_features_pretraining), perm=[1, 0, 2]).get_shape())
    print(self._input_sample_data.get_shape())
    self.rnn_pretraining_loss = tf.reduce_mean(tf.math.squared_difference(x=tf.transpose(tf.stack(self._generated_features_pretraining), perm=[1, 0, 2]), y=self._input_sample_data))
    if not FLAGS.disable_l2_regularizer:
      self.rnn_pretraining_loss = self.rnn_pretraining_loss+reg_loss
    
    pretraining_grads, _ = tf.clip_by_global_norm(tf.gradients(self.rnn_pretraining_loss, self.g_params), FLAGS.max_grad_norm)
    self.opt_pretraining = g_optimizer.apply_gradients(zip(pretraining_grads, self.g_params))

    # ---END, PRETRAINING---

    # The discriminator tries to tell the difference between samples from the
    # true data distribution (self.x) and the generated samples (self.z).
    #
    # Here we create two copies of the discriminator network (that share parameters),
    # as you cannot use the same network with different inputs in TensorFlow.
    with tf.compat.v1.variable_scope('D') as scope:
      scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
      # Make list of tensors. One per step in recurrence.
      # Each tensor is batchsize*numfeatures.
      # TODO: (possibly temporarily) disabling meta info
      print('self.sample_data shape {}'.format(self._input_sample_data.get_shape()))
      print('generated data shape {}'.format(self._generated_features[0].get_shape()))
      # TODO: (possibly temporarily) disabling meta info
      if FLAGS.generate_meta:
        sample_data_inputs = [tf.concat([self._input_metadata, sample_data_input],1) for sample_data_input in sample_data_inputs]
     
      self.real_d,self.real_d_features = self.discriminator(sample_data_inputs, is_training, msg='real')
      scope.reuse_variables()
      # TODO: (possibly temporarily) disabling meta info
      if FLAGS.generate_meta:
        generated_data = [tf.concat([meta_probs, sample_data_input],1) for sample_data_input in self._generated_features]
      else:
        generated_data = self._generated_features
      if sample_data_inputs[0].get_shape() != generated_data[0].get_shape():
        print('sample_data_inputs shape {} != generated data shape {}'.format(sample_data_inputs[0].get_shape(), generated_data[0].get_shape()))
      self.generated_d,self.generated_d_features = self.discriminator(generated_data, is_training, msg='generated')

    # Define the loss for discriminator and generator networks (see the original
    # paper for details), and create optimizers for both
    self.d_loss = tf.reduce_mean(-tf.math.log(tf.clip_by_value(self.real_d, 1e-1000000, 1.0)) \
                                 -tf.math.log(1 - tf.clip_by_value(self.generated_d, 0.0, 1.0-1e-1000000)))
    self.g_loss_feature_matching = tf.reduce_sum(tf.math.squared_difference(self.real_d_features, self.generated_d_features))
    self.g_loss = tf.reduce_mean(-tf.math.log(tf.clip_by_value(self.generated_d, 1e-1000000, 1.0)))

    if not FLAGS.disable_l2_regularizer:
      self.d_loss = self.d_loss+reg_loss
      self.g_loss_feature_matching = self.g_loss_feature_matching+reg_loss
      self.g_loss = self.g_loss+reg_loss
    self.d_params = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('model/D/')]

    if not is_training:
      return

    d_optimizer = tf.compat.v1.train.GradientDescentOptimizer(self._lr*FLAGS.d_lr_factor)
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_params),
                                        FLAGS.max_grad_norm)
    self.opt_d = d_optimizer.apply_gradients(zip(d_grads, self.d_params))
    if FLAGS.feature_matching:
      g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss_feature_matching,
                                                       self.g_params),
                                        FLAGS.max_grad_norm)
    else:
      g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params),
                                        FLAGS.max_grad_norm)
    self.opt_g = g_optimizer.apply_gradients(zip(g_grads, self.g_params))

    self._new_lr = tf.compat.v1.placeholder(shape=[], name="new_learning_rate", dtype=data_type())
    self._lr_update = tf.compat.v1.assign(self._lr, self._new_lr)

  def discriminator(self, inputs, is_training, msg=''):
    # RNN discriminator:
    
    if is_training and FLAGS.keep_prob < 1:
      inputs = [tf.nn.dropout(input_, rate=1-FLAGS.keep_prob) for input_ in inputs]      
    
    if is_training and FLAGS.keep_prob < 1:
      cell_fw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d,dropout_keep_prob=FLAGS.keep_prob)
      
      cell_bw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d,dropout_keep_prob=FLAGS.keep_prob)
    else:
      cell_fw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d)
      
      cell_bw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d)

    self._initial_state_fw = cell_fw.zero_state(self.batch_size, data_type())
    if not FLAGS.unidirectional_d:    
      self._initial_state_bw = cell_bw.zero_state(self.batch_size, data_type())
      print("cell_fw",cell_fw.output_size)

      outputs, state_fw, state_bw = tf.compat.v1.nn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=self._initial_state_fw, initial_state_bw=self._initial_state_bw)

    else:
      outputs, state = tf.nn.static_rnn(cell_fw, inputs, initial_state=self._initial_state_fw)

    if FLAGS.minibatch_d:
      outputs = [minibatch(tf.reshape(outp, shape=[FLAGS.batch_size, -1]), msg=msg, reuse_scope=(i!=0)) for i,outp in enumerate(outputs)]
    
    if FLAGS.end_classification:
      decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i!=0))) for i,output in enumerate([outputs[0], outputs[-1]])]
      decisions = tf.stack(decisions)
      decisions = tf.transpose(decisions, perm=[1,0,2])
      print('shape, decisions: {}'.format(decisions.get_shape()))
    else:
      decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i!=0))) for i,output in enumerate([outputs[0], outputs[-1]])]
      decisions = tf.stack(decisions)
      decisions = tf.transpose(decisions, perm=[1,0,2])
      # print('shape, decisions: {}'.format(decisions.get_shape()))
    decision = tf.reduce_mean(decisions, reduction_indices=[1,2])
    return (decision,tf.transpose(tf.stack(outputs), perm=[1,0,2]))
 
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def generated_features(self):
    return self._generated_features

  @property
  def input_sample_data(self):
    return self._input_sample_data

  @property
  def input_metadata(self):
    return self._input_metadata

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, loader, datasetlabel, eval_op_g, eval_op_d, pretraining=False, verbose=False, run_metadata=None, pretraining_d=False):
  """Runs the model on the given data."""
  epoch_start_time = time.time()
  g_loss, d_loss = 10.0, 10.0
  g_losses, d_losses = 0.0, 0.0
  iters = 0
  
  time_before_graph = None
  time_after_graph = None
  times_in_graph = []
  times_in_python = []
  
  loader.rewind(part=datasetlabel)
  [batch_meta, batch_data] = loader.get_batch(model.batch_size, part=datasetlabel)
  run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
  while batch_meta is not None and batch_data is not None:
    op_g = eval_op_g
    op_d = eval_op_d
    if datasetlabel == 'train' and not pretraining: # and not FLAGS.feature_matching:
      if d_loss == 0.0 and g_loss == 0.0:
        print('Both G and D train loss are zero. Exiting.')
        break
        #saver.save(session, checkpoint_path, global_step=m.global_step)
        #break
      elif d_loss == 0.0:
        #print('D train loss is zero. Freezing optimization. G loss: {:.3f}'.format(g_loss))
        op_g = tf.no_op()
      elif g_loss == 0.0: 
        #print('G train loss is zero. Freezing optimization. D loss: {:.3f}'.format(d_loss))
        op_d = tf.no_op()
      elif g_loss < 2.0 or d_loss < 2.0:
        if g_loss*.7 > d_loss:
          print('G train loss is {:.3f}, D train loss is {:.3f}. Freezing optimization of D'.format(g_loss, d_loss))
          op_g = tf.no_op()
        #elif d_loss*.7 > g_loss:
          #print('G train loss is {:.3f}, D train loss is {:.3f}. Freezing optimization of G'.format(g_loss, d_loss))
        op_d = tf.no_op()
    
    if pretraining:
      if pretraining_d:
        fetches = [model.rnn_pretraining_loss, model.d_loss, op_g, op_d]
      else:
        fetches = [model.rnn_pretraining_loss, tf.no_op(), op_g, op_d]
    else:
      fetches = [model.g_loss, model.d_loss, op_g, op_d]
    feed_dict = {}
    feed_dict[model.input_sample_data.name] = batch_data
    feed_dict[model.input_metadata.name] = batch_meta
    
    time_before_graph = time.time()
    if iters > 0:
      times_in_python.append(time_before_graph-time_after_graph)
    if run_metadata:
      g_loss, d_loss, _, _ = session.run(fetches, feed_dict, options=run_options, run_metadata=run_metadata)
    else:
      g_loss, d_loss, _, _ = session.run(fetches, feed_dict)
    time_after_graph = time.time()
    if iters > 0:
      times_in_graph.append(time_after_graph-time_before_graph)
    g_losses += g_loss
    if not pretraining:
      d_losses += d_loss
    iters += 1

    if verbose and iters % 10 == 9:
      # songs_per_sec = float(iters * model.batch_size)/float(time.time() - epoch_start_time)
      # avg_time_in_graph = float(sum(times_in_graph))/float(len(times_in_graph))
      # avg_time_in_python = float(sum(times_in_python))/float(len(times_in_python))
      #avg_time_batchreading = float(sum(times_in_batchreading))/float(len(times_in_batchreading))
      if pretraining:
        print("{}: {} (pretraining) batch loss: G: {:.3f}, avg loss: G: {:.3f}".format(datasetlabel, iters, g_loss, float(g_losses)/float(iters)))
      else:
        print("{}: {} batch loss: G: {:.3f}, D: {:.3f}, avg loss: G: {:.3f}, D: {:.3f}".format(datasetlabel, iters, g_loss, d_loss, float(g_losses)/float(iters), float(d_losses)/float(iters)))
    #batchtime = time.time()
    [batch_meta, batch_data] = loader.get_batch(model.batch_size,part=datasetlabel)
    #times_in_batchreading.append(time.time()-batchtime)

  if iters == 0:
    print("Iters == 0")
    return (None,None)

  g_mean_loss = g_losses/iters
  if pretraining and not pretraining_d:
    d_mean_loss = None
  else:
    d_mean_loss = d_losses/iters
  return (g_mean_loss, d_mean_loss)

def sample2(session, model, batch=False):
  """Samples from the generative model."""
  #state = session.run(model.initial_state)
  fetches = [model.generated_features]
  feed_dict = {}
  generated_features_list = []
  for i in range(int(FLAGS.n_samples/FLAGS.batch_size)+1):
    generated_features, = session.run(fetches, feed_dict)
    generated_features_list.append(generated_features)
    # fetches = [[tf.constant(generated_features[j], dtype=data_type()) for j in range(len(generated_features))]]
  

  # The following worked when batch_size=1.
  # generated_features = [np.squeeze(x, axis=0) for x in generated_features]
  # If batch_size != 1, we just pick the first sample. Wastefull, yes.
  returnable = []
  if batch:
    for generated_features in generated_features_list:
      for batchno in range(generated_features[0].shape[0]):
        returnable.append([x[batchno,:] for x in generated_features])
  else:
    returnable = [x[0,:] for x in generated_features]
  return returnable


def sample(session, model, batch=False):
  """Samples from the generative model."""
  #state = session.run(model.initial_state)
  fetches = [model.generated_features]
  feed_dict = {}
  generated_features_list = []
  for i in range(int(FLAGS.n_samples/FLAGS.batch_size)+1):
    generated_features, = session.run(fetches, feed_dict)
    generated_features_list.append(generated_features)
    fetches = [[tf.constant(generated_features[j], dtype=data_type()) for j in range(len(generated_features))]]
  

  # The following worked when batch_size=1.
  # generated_features = [np.squeeze(x, axis=0) for x in generated_features]
  # If batch_size != 1, we just pick the first sample. Wastefull, yes.
  returnable = []
  if batch:
    for generated_features in generated_features_list:
      for batchno in range(generated_features[0].shape[0]):
        returnable.append([x[batchno,:] for x in generated_features])
  else:
    returnable = [x[0,:] for x in generated_features]
  return returnable

def save_samples(i,session,m,generated_data_dir,experiment_label,global_step):
  song_data = []
  print ("Sampling from generator")

  
  song_data = sample(session, m, batch=True)
  song_data = np.array(song_data)
  song_data = song_data[:FLAGS.n_samples]

  # song_data2 = sample2(session, m, batch=True)
  # song_data2 = np.array(song_data2)
  # song_data2 = song_data2[:FLAGS.n_samples]


  
  shp = song_data.shape
  if(FLAGS.sample):
    # TODO: add exp name on synth data
    print ("Saving sample for epoch {} ...".format(global_step-1))    
    filename = os.path.join(generated_data_dir, 'sample_data_{}_{}.npy'.format(experiment_label, i))
    # filename1 = os.path.join(generated_data_dir, 'sample_data2_{}_{}_{}.npy'.format(experiment_label, global_step-1, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
  else:
    print ("Saving sample for epoch {} ...".format(i))
    filename = os.path.join(generated_data_dir, 'sample_data_{}_{}_{}.npy'.format(experiment_label, i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))   
    # filename1 = os.path.join(generated_data_dir, 'sample_data2_{}_{}_{}.npy'.format(experiment_label, i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))   
  np.save(filename,song_data)
  # np.save(filename1,song_data2)

  return song_data


def main(_):
  if not FLAGS.datadir:
    raise ValueError("Must set --datadir to datasets.")
  if not FLAGS.traindir:
    raise ValueError("Must set --traindir to dir where I can save model and plots.")
  if not FLAGS.filename:
    raise ValueError("Must set --filename to filename of dataset.")
  if not FLAGS.n_samples:
    raise ValueError("Must set --n_samples to the number of samples in the dataset.")
  if not FLAGS.timesteps:
    raise ValueError("Must set --timesteps to the timesteps in which the dataset is divided.")
  if not FLAGS.n_features:
    raise ValueError("Must set --n_features to number of features (variables in the dataset).")

  restore_flags()
 
  summaries_dir = None
  plots_dir = FLAGS.traindir+"/evaluations/plots"

  generated_data_dir = FLAGS.traindir+"/synthdata"
  summaries_dir = os.path.join(FLAGS.traindir, 'models/summaries')
  # plots_dir = os.path.join(FLAGS.traindir, 'plots')
  # generated_data_dir = os.path.join(FLAGS.traindir, 'generated_data')
  try: os.makedirs(FLAGS.traindir)
  except: pass
  try: os.makedirs(summaries_dir)
  except: pass
  # try: os.makedirs(plots_dir)
  # except: pass
  try: os.makedirs(generated_data_dir)
  except: pass
  directorynames = FLAGS.traindir.split('/')
  experiment_label = ''
  while not experiment_label:
    experiment_label = directorynames.pop()
  
  global_step = -1
  if os.path.exists(os.path.join(FLAGS.traindir, 'global_step.pkl')):
    with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'r') as f:
      global_step = pkl.load(f)
  global_step += 1
  # global_step = 294
  print("global step",global_step) 
  songfeatures_filename = os.path.join(FLAGS.traindir, 'num_song_features.pkl')
  metafeatures_filename = os.path.join(FLAGS.traindir, 'num_meta_features.pkl')
  synthetic=None
  if FLAGS.synthetic_chords:
    synthetic = 'chords'
    print('Training on synthetic chords!')
  if FLAGS.composer is not None:
    print('Single composer: {}'.format(FLAGS.composer))
  #loader = music_data_utils.MusicDataLoader(FLAGS.datadir, FLAGS.select_validation_percentage, FLAGS.select_test_percentage, FLAGS.works_per_composer, FLAGS.pace_events, synthetic=synthetic, tones_per_cell=FLAGS.tones_per_cell, single_composer=FLAGS.composer)
  loader = data_loader.DataLoader(FLAGS.datadir,FLAGS.select_validation_percentage,FLAGS.select_test_percentage,filename=FLAGS.filename,
                                  n_samples=FLAGS.n_samples,n_features=FLAGS.n_features,n_steps=FLAGS.timesteps,scale=FLAGS.scale)
  
  # if FLAGS.synthetic_chords:
  #   # This is just a print out, to check the generated data.
  #   batch = loader.get_batch(batchsize=1, timesteps=400)
  #   loader.get_midi_pattern([batch[1][0][i] for i in xrange(batch[1].shape[1])])

  # scaler = MinMaxScaler().fit(loader.data.reshape(-1,1))
  # p.save_plot_sample(loader.data,-1,FLAGS.testname,plots_dir,rsample=True)
  num_features = loader.get_num_features()
  print('num_song_features:{}'.format(num_features))
  num_meta_features = loader.get_num_meta_features()
  print('num_meta_features:{}'.format(num_meta_features))
  print(FLAGS.sample)

  train_start_time = time.time()
  checkpoint_path = os.path.join(FLAGS.traindir, "models/model.ckpt")

  timesteps_ceiling = FLAGS.timesteps
 
  with tf.Graph().as_default(), tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as session:
    with tf.compat.v1.variable_scope("model", reuse=None) as scope:
      scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
      m = RNNGAN(is_training=True, num_song_features=num_features, num_meta_features=num_meta_features,session=session)

    # default: FALSE
    if FLAGS.initialize_d:
      vars_to_restore = {}
      for v in tf.trainable_variables():
        if v.name.startswith('model/G/'):
          print(v.name[:-2])
          vars_to_restore[v.name[:-2]] = v
      saver = tf.train.Saver(vars_to_restore)
      ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
      if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path,end=" ")
        saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.initialize_variables([v for v in tf.trainable_variables() if v.name.startswith('model/D/')]))
      else:
        print("Created model with fresh parameters.")
        session.run(tf.compat.v1.global_variables_initializer())
      saver = tf.train.Saver(tf.compat.v1.global_variables())
    else:
      saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
      ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
      if ckpt:
        ckpt_path = "{}.*".format(ckpt.model_checkpoint_path)
        if (glob(ckpt_path)):
          print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
          saver.restore(session, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        session.run(tf.compat.v1.global_variables_initializer())
    
    run_metadata = None
    # default: False
    if FLAGS.profiling:
      run_metadata = tf.RunMetadata()
    # default: False
    if not FLAGS.sample: # if is training
      print ("Training...")
      train_g_loss,train_d_loss = 1.0,1.0
      for i in range(global_step, FLAGS.max_epoch):
        lr_decay = FLAGS.lr_decay ** max(i - FLAGS.epochs_before_decay, 0.0)
        """
        Change the timesteps. Not useful for our case
        """
      
        if not FLAGS.adam:
          m.assign_lr(session, FLAGS.learning_rate * lr_decay)

        save = False
        do_exit = False

        print("Epoch: {} Learning rate: {:.3f}, pretraining: {}".format(i, session.run(m.lr), (i<FLAGS.pretraining_epochs)))
        if i<FLAGS.pretraining_epochs:          
          opt_d = tf.no_op()
          if FLAGS.pretraining_d:
            opt_d = m.opt_d          
          train_g_loss,train_d_loss = run_epoch(session, m, loader, 'train', m.opt_pretraining, opt_d, pretraining = True, verbose=True, run_metadata=run_metadata, pretraining_d=FLAGS.pretraining_d)
          if FLAGS.pretraining_d:
            try:
              print("Epoch: {} Pretraining loss: G: {:.3f}, D: {:.3f}".format(i, train_g_loss, train_d_loss))
            except:
              print(train_g_loss)
              print(train_d_loss)
          else:
            print(train_g_loss)
            print("Epoch: {} Pretraining loss: G: {:.3f}".format(i, train_g_loss))
        else:
          train_g_loss,train_d_loss = run_epoch(session, m, loader, 'train', m.opt_d, m.opt_g, verbose=True, run_metadata=run_metadata)
          try:
            print("Epoch: {} Train loss: G: {:.3f}, D: {:.3f}".format(i, train_g_loss, train_d_loss))
          except:
            print("Epoch: {} Train loss: G: {}, D: {}".format(i, train_g_loss, train_d_loss))
        
        
        if(FLAGS.check_on_valid):
          valid_g_loss,valid_d_loss = run_epoch(session, m, loader, 'validation', tf.no_op(), tf.no_op())
          try:
            print("Epoch: {} Valid loss: G: {:.3f}, D: {:.3f}".format(i, valid_g_loss, valid_d_loss))
          except:
            print("Epoch: {} Valid loss: G: {}, D: {}".format(i, valid_g_loss, valid_d_loss))
        
        if train_d_loss == 0.0 and train_g_loss == 0.0:
          print('Both G and D train loss are zero. Exiting.')
          save = True
          do_exit = True
        if i % FLAGS.epochs_per_checkpoint == 0:
          save = True
        if FLAGS.exit_after > 0 and time.time() - train_start_time > FLAGS.exit_after*60:
          print("%s: Has been running for %d seconds. Will exit (exiting after %d minutes)."%(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), (int)(time.time() - train_start_time), FLAGS.exit_after))
          save = True
          do_exit = True

        if save:
          print("Saving checkpoint at {}".format(i))
          saver.save(session, checkpoint_path, global_step=i)
          with open(os.path.join(FLAGS.traindir, 'models/global_step.pkl'), 'wb') as f:
            pkl.dump(i, f)
          if FLAGS.profiling:
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(os.path.join(plots_dir, 'timeline.json'), 'w') as f:
              f.write(ctf)
          
          # s1 = save_samples(i,session,m,generated_data_dir,experiment_label,global_step)
          # ix = "{}".format(i)
          # p.save_plot_sample(s1,ix,FLAGS.testname,plots_dir)
          # ix = ix+"_2"
          # p.save_plot_sample(s2,ix,FLAGS.testname,plots_dir)
          # print('{}: Saving done!'.format(i))

        step_time, loss = 0.0, 0.0
        if train_d_loss is None: #pretraining
          train_d_loss = 0.0
          valid_d_loss = 0.0
          valid_g_loss = 0.0
        if not os.path.exists(os.path.join(plots_dir, 'gnuplot-input.txt')):
          print ("PLOTS DIR", plots_dir)
          with open(os.path.join(plots_dir, 'gnuplot-input.txt'), 'w') as f:
            f.write('# global-step learning-rate train-g-loss train-d-loss valid-g-loss valid-d-loss\n')
        with open(os.path.join(plots_dir, 'gnuplot-input.txt'), 'a') as f:
          try:
            if(FLAGS.check_on_valid):
              f.write('{} {:.4f} {:.2f} {:.2f} {:.3} {:.3f}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))
            else:
              f.write('{} {:.4f} {:.2f} {:.2f}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss))
          except:
            if(FLAGS.check_on_valid):             
              f.write('{} {} {} {} {} {}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))
            else:
              f.write('{} {} {} {}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss))
        if not os.path.exists(os.path.join(plots_dir, 'gnuplot-commands-loss.txt')):
          with open(os.path.join(plots_dir, 'gnuplot-commands-loss.txt'), 'a') as f:
            f.write('set terminal postscript eps color butt "Times" 14\nset yrange [0:400]\nset output "loss.eps"\nplot \'gnuplot-input.txt\' using ($1):($3) title \'train G\' with linespoints, \'gnuplot-input.txt\' using ($1):($4) title \'train D\' with linespoints, \'gnuplot-input.txt\' using ($1):($5) title \'valid G\' with linespoints, \'gnuplot-input.txt\' using ($1):($6) title \'valid D\' with linespoints, \n')
        
          
        if do_exit:
          if FLAGS.call_after is not None:
            print("%s: Will call \"%s\" before exiting."%(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), FLAGS.call_after))
            res = call(FLAGS.call_after.split(" "))
            print ('{}: call returned {}.'.format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), res))
          exit()
        sys.stdout.flush()

      # TODO descomentar essa parte
      #test_g_loss,test_d_loss = run_epoch(session, m, loader, 'test', tf.no_op(), tf.no_op())
      #print("Test loss G: %.3f, D: %.3f" %(test_g_loss, test_d_loss))
    gc.collect()
    if FLAGS.sample:
      # i = 0
      for i in range(FLAGS.n_synth_sample):
        print ("Saving {}".format(i))
        save_samples(i,session,m,generated_data_dir,experiment_label,global_step)
    
    
if __name__ == "__main__":
  tf.compat.v1.app.run()