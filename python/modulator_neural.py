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

import numpy
import time
import uuid
from gnuradio import gr
import pmt
import torch
from torch_echo.modulators import ModulatorNeural
from torch_echo.utils.util_data import integers_to_symbols, bits_to_integers

from torch_echo.utils.visualize import get_constellation, visualize_constellation
import matplotlib.pyplot as plt


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
"""


class modulator_neural(gr.basic_block, ModulatorNeural):
    """
    Define neural net parameters, loss, optimizer, defaults to fast convergence
    Inputs:
    seed: the tf.random seed to be used
    hidden_layers: np array of shape [m]/ list of length [m] containing number of units in each hidden layer
    bits_per_symbol: NN takes in input of this size
    preamble: pseudo-random sequence used to update the modulator after a round trip
    log_constellation_interval: number of updates between logging the mod constellation
    lambda_p: Scaling factor for power loss term (used only when restrict_energy is False)
    restrict_energy: If true normalize outputs(re + 1j*im) to have average energy 1
    std_min: Minimum standard deviation while exploring
    initial_logstd: Initial log standard deviation of exploring
    activation_fn_hidden: Activation function to be used for hidden layers (default = tf.nn.relu)
    kernel_initializer_hidden:  Kernel initializer for hidden layers (default = normc_initializer(1.0))
    bias_initializer_hidden: Bias initialize for hidden layers (default = tf.glorot_uniform_initializer())
    activation_fn_output: Activation function to be used for output layer (default = None)
    kernel_initializer_output: Kernel initializer for output layer (default = normc_initializer(1.0))
    bias_initializer_output: Bias initializer for output layer (default = tf.glorot_uniform_initializer())
    optimizer: Optimizer to be used while training (default = tf.train.AdamOptimizer),
    """
    def __init__(self, seed=189, hidden_layers=(64,), bits_per_symbol=2,
                 preamble=None, log_constellation_interval=10,
                 init_weights="",
                 lambda_p=0.0,
                 max_std=1e1,
                 min_std=1e-2,
                 initial_std=4e-2,
                 restrict_energy=3,
                 activation_fn_hidden='tanh',
                 # kernel_initializer_hidden={'normc_std': 1.0},
                 # bias_initializer_hidden=None,
                 activation_fn_output=None,
                 # kernel_initializer_output=normc_initializer(1.0),
                 # bias_initializer_output=tf.glorot_uniform_initializer(),
                 optimizer='adam',
                 lambda_prob=1e-9,
                 stepsize_mu=1e-2,
                 stepsize_sigma=5e-4
                 ):
        gr.basic_block.__init__(self,
                                name="modulator_neural",
                                in_sig=None,
                                out_sig=None)
        ModulatorNeural.__init__(self,
                                 seed=seed, hidden_layers=hidden_layers,
                                 bits_per_symbol=bits_per_symbol,
                                 lambda_p=lambda_p,
                                 max_std=max_std,
                                 min_std=min_std,
                                 initial_std=initial_std,
                                 restrict_energy=restrict_energy,
                                 activation_fn_hidden=activation_fn_hidden,
                                 # kernel_initializer_hidden=kernel_initializer_hidden,
                                 # bias_initializer_hidden=bias_initializer_hidden,
                                 activation_fn_output=activation_fn_output,
                                 # kernel_initializer_output=kernel_initializer_output,
                                 # bias_initializer_output=bias_initializer_output,
                                 optimizer=optimizer,
                                 lambda_prob=lambda_prob,
                                 stepsize_mu=stepsize_mu,
                                 stepsize_sigma=stepsize_sigma)
        if preamble is None:
            raise Exception("Preamble must be provided")
        if preamble is not numpy.ndarray:
            preamble = numpy.array(preamble)
        assert len(preamble.shape) == 1, "Preamble must be a vector, not a matrix with a dimension of size 1"
        self.preamble = preamble
        self.preamble_si = bits_to_integers(self.preamble, self.bits_per_symbol)
        self.preamble_len = len(self.preamble)

        self.run_mode = "train"
        self.init_weights = init_weights
        if len(init_weights) > 0:
            self.model.load_state_dict(torch.load(init_weights))
            self.run_mode = "freeze"
        # Message ports
        self.port_id_in = pmt.intern("bits")
        self.port_id_out = pmt.intern("symbols")
        self.port_id_feedback = pmt.intern("feedback")
        self.port_id_ctrl = pmt.intern("control")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_in(self.port_id_feedback)
        self.message_port_register_in(self.port_id_ctrl)
        self.message_port_register_out(self.port_id_out)
        self.set_msg_handler(self.port_id_in, self.handle_packet)
        self.set_msg_handler(self.port_id_feedback, self.handle_feedback)
        self.set_msg_handler(self.port_id_ctrl, self.handle_control)
        # Meta variables
        self.packet_cnt = 0
        self.train_cnt = 0
        self.version = 0
        self.version_dirty = False
        self.past_actions = None

        self.log_constellation_interval = log_constellation_interval
        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("neural mod {}: {} bits per symbol".format(self.uuid_str, self.bits_per_symbol))
        if len(init_weights) > 0:
            self.logger.info("neural mod {}: initialized weights from {}".format(self.uuid_str, init_weights))
        with open("ber_echo_{}.csv".format(self.uuid_str), "w") as f:
            f.write("train_iter,BER\n")

    @staticmethod
    def center_iq(symbs):
        # Move the symbols toward the center to reduce IQ imbalance issues
        center = numpy.mean(symbs)
        symbs = symbs - center
        # Clip the symbols to the unit circle for USRP transmission
        abssymbs = numpy.abs(symbs)
        symbs[abssymbs > 1] /= abssymbs[abssymbs > 1]
        # Move the symbols toward the center to reduce IQ imbalance issues
        center = numpy.mean(symbs)
        symbs = symbs - center
        return symbs

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
        # Check for keys we care about
        if "freeze" in keys:
            self.run_mode = "freeze"
            self.logger.info("neural mod {}: freezing model and saving state".format(self.uuid_str))
            torch.save(self.model.state_dict(), "mod_neural_weights-{}".format(self.uuid_str))
            print("Saving final mod constellation @{}".format(self.train_cnt))
            numpy.save("neural_mod_constellation_{:05d}_{}".format(self.train_cnt,
                       time.strftime('%Y%m%d_%H%M%S')),
                       get_constellation(self))
        elif "train" in keys:
            self.logger.info("neural mod {}: resuming training".format(self.uuid_str))
            self.run_mode = "train"

    def handle_packet(self, pdu):
        t0 = time.time()
        if self.version_dirty:
            pass  # Some packets may be dropped, need to keep going
        self.packet_cnt += 1
        tag_dict = pmt.to_python(pmt.car(pdu))
        vec = pmt.to_python(pmt.cdr(pdu))
        if self.run_mode == "freeze":
            symbs = self.modulate(bits_to_integers(vec, self.bits_per_symbol), 'exploit')
        else:
            symbs = self.modulate(bits_to_integers(vec, self.bits_per_symbol), 'explore')
        # Condition the symbols to reduce IQ imbalance issues
        symbs = self.center_iq(symbs)
        # self.logger.info("symbs[{}] {}: {}".format(type(symbs[0]), symbs.shape, symbs))
        self.past_actions = symbs
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs.astype(numpy.complex64))))
        self.version_dirty = True
        t1 = time.time()
        self.logger.debug("neural mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def handle_feedback(self, pdu):
        t0 = time.time()
        self.train_cnt += 1
        tag_dict = pmt.to_python(pmt.car(pdu))
        vec = pmt.to_python(pmt.cdr(pdu))
        # Split packet into functional parts
        new_echo = vec[:self.preamble.size]
        my_echo = vec[self.preamble.size:2 * self.preamble.size]
        # rest_of_packet = vec[2 * self.preamble.size:]
        # Do not perform updates while frozen
        if self.run_mode == "freeze":
            # Save every round trip BER while frozen to measure average performance
            ber = sum(numpy.abs(self.preamble - my_echo)) * 1.0 / self.preamble.size
            with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                f.write("{},{}\n".format(self.train_cnt, ber))
        else:  # run_mode == train
            labels_si_g = bits_to_integers(my_echo, self.bits_per_symbol)
            # Only use preamble from feedback, at least for now
            reward, _, _, loss = self.update(preamble_si=self.preamble_si,  # labels
                                             actions=self.past_actions[0:self.preamble_si.shape[0]],  # actions
                                             labels_si_g=labels_si_g[0:self.preamble_si.shape[0]])  # demodulator guesses
            self.version_dirty = False
            if self.train_cnt % self.log_constellation_interval == 0:
                print("Saving mod constellation @{}".format(self.train_cnt))
                numpy.save("neural_mod_constellation_{:05d}_{}".format(self.train_cnt,
                           time.strftime('%Y%m%d_%H%M%S')),
                           get_constellation(self))
                ber = sum(numpy.abs(self.preamble - my_echo)) * 1.0 / self.preamble.size
                with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(self.train_cnt, ber))
        # Send the echo with a new preamble to continue the training cycle
        echo_packet = numpy.concatenate([self.preamble, new_echo])
        echo_packet_si = bits_to_integers(echo_packet, self.bits_per_symbol)
        exploration = 'exploit' if self.run_mode == 'freeze' else 'explore' 
        echo = self.modulate(echo_packet_si, exploration)
        # Condition the symbols to reduce IQ imbalance issues
        echo = self.center_iq(echo)
        self.past_actions = echo
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(echo.astype(numpy.complex64))))
        self.version_dirty = True
        # Logging
        t1 = time.time()
        self.logger.debug("neural mod {} updated with {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def update_version(self):
        """
        Increment the version counter
        """
        self.version += 1
        return self.version
