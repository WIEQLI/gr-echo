#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Josh Sanz <jsanz@berkeley.edu>
# 2019 09 13
#
# Copyright 2018 <+YOU OR YOUR COMPANY+>.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from __future__ import print_function
import numpy
import time
import uuid
import torch
from gnuradio import gr
import pmt
from torch_echo.demodulators import DemodulatorNeural
# from torch_echo.utils.util_tf import normc_initializer
from torch_echo.utils.util_data import integers_to_bits, bits_to_integers
from torch_echo.utils.visualize import gen_demod_grid


"""
neural_mod_qpsk_vs_clone_simple={
    'seed':189,
    'hidden_layers': [50],
    'kernel_initializer': {
        'normc_std':1.0,
    },
    'activation_fn_hidden':'tanh',
    'stepsize_mu':     1e-3,
    'stepsize_sigma':   5e-4,
    'initial_std':      4e-1,
    'min_std': 1e-2,
    'max_std':100,
    'lambda_prob':1e-10,
    'restrict_energy':True,
    'lambda_p': 0.0
}
neural_demod_qpsk_vs_clone_simple={
    'seed':37, #PLACEHOLDER
    'hidden_layers': [50],
    'kernel_initializer':  {
        'normc_std':1.0,
    },
    'activation_fn_hidden':'tanh',
    'loss_type':'l2',
    'stepsize_cross_entropy': 1e-3,
    'cross_entropy_weight': 1.0
}
"""


class demodulator_neural(gr.basic_block, DemodulatorNeural):
    """
    Inputs:
    seed: Tensor flow graph level seed (default = 7)
    hidden_layers: A list of length [num_layers] with entries corresponding to
                   number of hidden units in each layer (Default = [16])
    bits_per_symbol: Determines number of units in output layer as 2**bits_per_symbol
    preamble: pseudo-random sequence used to update the demodulator
    log_constellation_interval: number of updates between logging the demod constellation
    activation_fn_hidden: Activation function to be used for hidden layers (default = tf.nn.relu)
    kernel_initializer_hidden:  Kernel initializer for hidden layers (default = normc_initializer(1.0))
    bias_initializer_hidden: Bias initialize for hidden layers (default = tf.glorot_uniform_initializer())
    activation_fn_output: Activation function to be used for output layer (default = None)
    kernel_initializer_output: Kernel initializer for output layer (default = normc_initializer(1.0))
    bias_initializer_output: Bias initializer for output layer (default = tf.glorot_uniform_initializer())
    optimizer: Optimizer to be used while training (default = tf.train.AdamOptimizer),
    initial_eps: Initial probability for exploring each class
    min_eps: Minimum probability for exploring each class
    max_eps: Maximum probability for exploring each class
    lambda_prob: Regularizer for log probability
    """

    def __init__(self, seed=0, hidden_layers=(64,), bits_per_symbol=2,
                 preamble=None, log_constellation_interval=10,
                 init_weights="",
                 activation_fn_hidden='tanh',
                 # kernel_initializer_hidden=normc_initializer(1.0),
                 # bias_initializer_hidden=tf.glorot_uniform_initializer(),
                 activation_fn_output=None,
                 # kernel_initializer_output=normc_initializer(1.0),
                 # bias_initializer_output=tf.glorot_uniform_initializer(),
                 optimizer=torch.optim.Adam,
                 # initial_eps=1e-1,
                 # max_eps=2e-1,
                 # min_eps=1e-4,
                 lambda_prob=1e-10,
                 loss_type='l2',
                 # explore_prob=0.5,
                 # strong_supervision_prob=0.,
                 stepsize_mu=1e-3,
                 # stepsize_eps=1e-5,
                 stepsize_cross_entropy=1e-3,
                 cross_entropy_weight=1.0,
                 ):
        gr.basic_block.__init__(self,
                                name="demodulator_neural",
                                in_sig=None,
                                out_sig=None)
        DemodulatorNeural.__init__(self,
                                   seed=seed, hidden_layers=hidden_layers,
                                   bits_per_symbol=bits_per_symbol,
                                   activation_fn_hidden=activation_fn_hidden,
                                   # kernel_initializer_hidden=kernel_initializer_hidden,
                                   # bias_initializer_hidden=bias_initializer_hidden,
                                   activation_fn_output=activation_fn_output,
                                   # kernel_initializer_output=kernel_initializer_output,
                                   # bias_initializer_output=bias_initializer_output,
                                   optimizer=optimizer,
                                   # initial_eps=initial_eps,
                                   # max_eps=max_eps,
                                   # min_eps=min_eps,
                                   lambda_prob=lambda_prob,
                                   loss_type=loss_type,
                                   # explore_prob=explore_prob,
                                   # strong_supervision_prob=strong_supervision_prob,
                                   stepsize_mu=stepsize_mu,
                                   # stepsize_eps=stepsize_eps,
                                   stepsize_cross_entropy=stepsize_cross_entropy,
                                   cross_entropy_weight=cross_entropy_weight)
        if preamble is None:
            raise Exception("You must provide a preamble")
        if preamble is not numpy.ndarray:
            preamble = numpy.array(preamble)
        assert len(preamble.shape) == 1, "Preamble must be a vector, not a matrix with a dimension of size 1"
        self.preamble = preamble
        self.preamble_si = bits_to_integers(numpy.array(self.preamble), self.bits_per_symbol)
        self.run_mode = "train"   # Be careful not to clobber the parent class' mode here!
        if len(init_weights) > 0:
            self.model.load_state_dict(torch.load(init_weights))
            self.run_mode = "freeze"
        # Message ports
        self.port_id_in = pmt.intern("symbols")
        self.port_id_out = pmt.intern("bits")
        self.port_id_ctrl = pmt.intern("control")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_in(self.port_id_ctrl)
        self.message_port_register_out(self.port_id_out)
        self.set_msg_handler(self.port_id_in, self.handle_packet)
        self.set_msg_handler(self.port_id_ctrl, self.handle_control)
        # Counters
        self.packet_cnt = 0
        self.log_constellation_interval = log_constellation_interval
        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("neural demod {}: {} bits per symbol".format(self.uuid_str, self.bits_per_symbol))
        if len(init_weights) > 0:
            self.logger.info("neural demod {}: initialized weights from {}".format(self.uuid_str, init_weights))
        with open("ber_{}.csv".format(self.uuid_str), "w") as f:
            f.write("iter,BER\n")

    def handle_control(self, msg):
        # Test whether message is a dict, per UHD requirements
        if not pmt.is_dict(msg):
            self.logger.info("{} received non-dict control message, ignoring".format(self.uuid_str))
            return
        try:
            keys = pmt.to_python(pmt.dict_keys(msg))
        except pmt.wrong_type as e:
            self.logger.debug("{} received pair instead of dict, fixing".format(self.uuid_str))
            msg = pmt.dict_add(pmt.make_dict(), pmt.car(msg), pmt.cdr(msg))
            keys = pmt.to_python(pmt.dict_keys(msg))
        print("KEYS: {}".format(keys))
        # Check for keys we care about
        if "freeze" in keys:
            self.run_mode = "freeze"
            self.logger.info("neural demod {}: freezing model and saving state".format(self.uuid_str))
            torch.save(self.model.state_dict(), "demod_neural_weights-{}".format(self.uuid_str))
            print("Saving final demod constellation @{}".format(self.packet_cnt))
            data_vis = gen_demod_grid(points_per_dim=40, min_val=-2.5, max_val=2.5)['data']
            labels_si_g = self.demodulate(data_c=data_vis, mode='exploit')
            numpy.savez("neural_demod_constellation_{:05d}_{}".format(self.packet_cnt,
                        time.strftime('%Y%m%d_%H%M%S')),
                        iq=data_vis, labels=labels_si_g)
        elif "train" in keys:
            self.run_mode = "train"
            self.logger.info("neural demod {}: resuming training".format(self.uuid_str))

    def handle_packet(self, pdu):
        t0 = time.time()
        self.packet_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.cdr(pdu)
        # tag_p = pmt.to_python(tag)
        data_c = pmt.to_python(vec)
        # update
        if self.run_mode == "train":
            self.train_preamble(data_c[0:len(self.preamble_si)])
        # demod
        labels_g = integers_to_bits(self.demodulate(data_c, 'exploit'), self.bits_per_symbol)
        # Sample some BERs during training, but sample all while frozen to collect average
        # behavior faster
        if self.packet_cnt % self.log_constellation_interval == 0 or self.run_mode == "freeze":
            ber = sum(numpy.abs(self.preamble - labels_g[:self.preamble.size])) * 1.0 / self.preamble.size
            with open("ber_{}.csv".format(self.uuid_str), "a") as f:
                f.write("{},{}\n".format(self.packet_cnt, ber))
        # publish
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(labels_g.astype(dtype=numpy.int8))))
        t1 = time.time()
        self.logger.debug("neural demod {} handled {} bits in {} seconds".format(
                          self.uuid_str, labels_g.size, t1 - t0))

    def train_preamble(self, data_c):
        self.update(inputs=self.preamble_si, actions=[], data_for_rewards=data_c, mode='echo')
        if self.packet_cnt % self.log_constellation_interval == 0:
            print("Saving demod constellation @{}".format(self.packet_cnt))
            data_vis = gen_demod_grid(points_per_dim=40, min_val=-2.5, max_val=2.5)['data']
            labels_si_g = self.demodulate(data_c=data_vis, mode='exploit')
            numpy.savez("neural_demod_constellation_{:05d}_{}".format(self.packet_cnt,
                        time.strftime('%Y%m%d_%H%M%S')),
                        iq=data_vis, labels=labels_si_g)
