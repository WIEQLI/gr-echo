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

from collections import OrderedDict
import random
import struct
import time
import uuid

from gnuradio import gr
import matplotlib.pyplot as plt
import numpy as np
import pmt
import reedsolo
import torch

from torch_echo.modulators import ModulatorNeural, ModulatorClassic
from torch_echo.utils.util_data import integers_to_symbols, bits_to_integers
###DEBUG###
from torch_echo.demodulators import DemodulatorClassic
###DEBUG###


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


def bytes_to_bits(input):
    return np.unpackbits(np.array(input)).astype(np.bool)


def bits_to_bytes(input):
    return bytearray(np.packbits(input))


class modulator_neural_spy(gr.basic_block, ModulatorNeural):
    """
    Define neural net parameters, loss, optimizer, defaults to fast convergence
    Inputs:
    seed: the tf.random seed to be used
    hidden_layers: np array of shape [m]/ list of length [m] containing number of units in each hidden layer
    bits_per_symbol: NN takes in input of this size
    preamble: pseudo-random sequence used to update the modulator after a round trip
    log_constellation_interval: number of updates between logging the mod constellation
    spy_length: number of bits used to 'spy' on packets to ensure that they aren't inordinately corrupted
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
    mod_types = {1: 'BPSK', 2: 'QPSK', 3: '8PSK', 4: 'QAM16', 6: 'QAM64'}

    def __init__(self, seed=189, hidden_layers=[64], bits_per_symbol=2,
                 preamble=None, log_constellation_interval=10,
                 spy_length=64, spy_threshold=0.1,
                 init_weights="",
                 lambda_p=0.0,
                 max_std=8e-1,
                 min_std=1e-1,
                 initial_std=2.0e-1,
                 restrict_energy=3,
                 activation_fn_hidden='tanh',
                 # kernel_initializer_hidden={'normc_std': 1.0},
                 # bias_initializer_hidden=None,
                 activation_fn_output=None,
                 # kernel_initializer_output=normc_initializer(1.0),
                 # bias_initializer_output=tf.glorot_uniform_initializer(),
                 optimizer='sgd',
                 lambda_prob=1e-10,
                 stepsize_mu=1e-1,
                 stepsize_sigma=1e-3
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
        ###DEBUG###
        self.seed = seed
        #self.model.load_state_dict(torch.load("pretrained.mdl"))
        torch.save(self.model.state_dict(), "initialization.mdl")
        torch.save(self.model.state_dict(), "model_0.mdl")
        np.save("const0", self.get_constellation())
        ###DEBUG###

        # Echo protocol variables
        if preamble is None:
            raise Exception("Preamble must be provided")
        if preamble is not np.ndarray:
            preamble = np.array(preamble)
        assert len(preamble.shape) == 1, "Preamble must be a vector, not a matrix with a dimension of size 1"
        self.preamble = preamble
        self.preamble_si = bits_to_integers(self.preamble, self.bits_per_symbol)
        self.preamble_len = len(self.preamble)
        np.save("preamble_si", self.preamble_si)
        self.past_actions = None

        self.run_mode = "train"
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

        # Spy and packet info
        self.spy_length = spy_length
        self.spy_threshold = spy_threshold
        assert self.spy_length % self.bits_per_symbol == 0
        self.spy_mod = ModulatorClassic(self.bits_per_symbol)
        # | 1byte valid flag | 2byte sequence number |
        self.reedsolomon = reedsolo.RSCodec(4)
        self.rs_length = 4 * 2 * 8  # 4 bytes data, 4 bytes parity, 8 bits per byte

        # Logging
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
        ###DEBUG###
        #self.neur_noise = np.load("neur_noise.npy", allow_pickle=True)
        #self.class_noise = np.load("class_noise.npy", allow_pickle=True)
        self.demod = DemodulatorClassic(self.bits_per_symbol)
        self.seeds = np.load("python-inputs/seeds.npy")
        ###DEBUG###

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
            np.save("neural_mod_constellation_{:05d}_{}".format(self.train_cnt,
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
        exploration = 'exploit' if self.run_mode == 'freeze' else 'explore' 
        symbs = self.modulate(bits_to_integers(vec, self.bits_per_symbol), exploration)
        self.past_actions = symbs
        # Spy and header fields
        classic, _ = self.assemble_packet(vec[self.preamble.size:], valid=False)
        classic_si = bits_to_integers(classic, self.bits_per_symbol)
        symbs = np.concatenate([self.spy_mod.modulate(classic_si), symbs])
        # self.logger.info("symbs[{}] {}: {}".format(type(symbs[0]), symbs.shape, symbs))
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs.astype(np.complex64))))
        self.version_dirty = True
        t1 = time.time()
        self.logger.debug("neural mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def handle_feedback(self, pdu):
        t0 = time.time()
        self.train_cnt += 1
        self.packet_cnt += 1
        tag_dict = pmt.to_python(pmt.car(pdu))
        vec = pmt.to_python(pmt.cdr(pdu))
        # Split packet into functional parts
        spy, hdr, new_echo, my_echo = self.split_packet(vec)
        spy_ber = sum(spy != self.preamble[:self.spy_length]) * 1.0 / self.spy_length
        if hdr is not None:
            valid = hdr[0]
            pktidx = hdr[1]
        else:
            valid = False
        # Do not perform updates while frozen
        if self.run_mode == "freeze" and spy_ber < self.spy_threshold and valid:
            # Save every round trip BER while frozen to measure average performance
            ber = sum(np.abs(self.preamble - my_echo)) * 1.0 / self.preamble.size
            with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                f.write("{},{}\n".format(self.train_cnt, ber))
        elif spy_ber < self.spy_threshold and valid:  # run_mode == "train"
            labels_si_g = bits_to_integers(my_echo, self.bits_per_symbol)
            labels_all = bits_to_integers(np.concatenate([new_echo, my_echo]), 
                                          self.bits_per_symbol)
            p_actions = self.past_actions[0:self.preamble_si.shape[0]]  # actions
            
            ###DEBUG###
            np.save("labels_si_g_{}".format(pktidx), labels_all)
            np.save("past_actions_{}".format(pktidx), self.past_actions)

            #labels_perfect = np.load("outputs.sgd.0/labels_si_g_{}.npy".format(self.train_cnt))

            labels_perfect = self.demod.demodulate(p_actions + (np.random.randn(p_actions.size) + 1j * np.random.randn(p_actions.size)) * 0.2)
            #errors = sum(np.abs(labels_perfect - labels_si_g))
            #print("labels diff", errors)
            #if errors < 300:
            #    labels_perfect = labels_si_g

            #nidx = (self.train_cnt-1)  % self.class_noise.shape[0]# np.random.choice(self.neur_noise.shape[0])
            #print(nidx, self.class_noise.shape[0])
            #np.save("saved_clrx_{}".format(self.train_cnt), p_actions + self.class_noise[nidx])
            #cllabels = self.demod.demodulate(p_actions + self.class_noise[nidx])
            #if np.random.random() < 0.5:
            #    cllabels = np.random.randint(0, 4, cllabels.size)
            #np.save("saved_nerx_{}".format(self.train_cnt), ModulatorClassic(2).modulate(cllabels) + self.neur_noise[nidx])
            #labels_perfect = self.demod.demodulate(ModulatorClassic(2).modulate(cllabels) + 
            #        self.neur_noise[nidx])
            #np.save("labels_si_g_{}".format(self.train_cnt), labels_perfect)

            ###DEBUG###

            ###DEBUG###
            # Load a static dataset and train on it instead
            #labels_si_g = np.load("python-inputs/labels_si_g_{}.input.npy".format(self.train_cnt - 1))
            #p_actions = np.load("python-inputs/past_actions_{}.input.npy".format(self.train_cnt - 1))
            #torch.manual_seed(self.seeds[self.train_cnt])
            #np.random.seed(self.seeds[self.train_cnt])
            #random.seed(self.seeds[self.train_cnt])
            ###DEBUG###

            # Only use preamble from feedback, at least for now
            torch.save(self.model.state_dict(), "tmp.mdl")
            reward, std0, std1, loss = self.update(preamble_si=self.preamble_si,  # labels
                                             actions=p_actions,
                                             labels_si_g=labels_si_g)  # demodulator guesses
                                             #labels_si_g=labels_perfect)  # demodulator guesses

            ###DEBUG###
            np.save("reward_{}".format(pktidx), reward)
            np.save("std0_{}".format(pktidx), std0)
            np.save("std1_{}".format(pktidx), std1)
            np.save("loss_{}".format(pktidx), loss)
            torch.save(self.model.state_dict(), "model_{}.mdl".format(pktidx))
            ###DEBUG###
            self.version_dirty = False
            if self.train_cnt % self.log_constellation_interval == 0:
                print("Saving mod constellation @{}".format(pktidx))
                np.save("neural_mod_constellation_{:05d}_{}".format(pktidx,
                           time.strftime('%Y%m%d_%H%M%S')),
                           self.get_constellation())
                ber = sum(self.preamble != my_echo) * 1.0 / self.preamble.size
                with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(pktidx, ber))
        # Send the echo with a new preamble to continue the training cycle
        classic, neural = self.assemble_packet(new_echo, valid=spy_ber < self.spy_threshold) 
        
        cl_si = bits_to_integers(classic, self.bits_per_symbol)
        cl_symb = self.spy_mod.modulate(cl_si)
       
        nn_si = bits_to_integers(neural, self.bits_per_symbol)
        exploration = 'exploit' if self.run_mode == 'freeze' else 'explore' 
        nn_symb = self.modulate(nn_si, exploration)
        self.past_actions = nn_symb
      
        packet = np.concatenate([cl_symb, nn_symb])
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(packet.astype(np.complex64))))
        self.version_dirty = True
        # Logging
        t1 = time.time()
        self.logger.debug("neural mod {} updated with {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def split_packet(self, bits):
        spy = bits[:self.spy_length]
        try:
            hdr = struct.unpack('BH', 
                    self.reedsolomon.decode(bits_to_bytes(
                        bits[self.spy_length:self.spy_length + self.rs_length])))
        except reedsolo.ReedSolomonError:
            hdr = None
        offset = self.spy_length + self.rs_length
        new_echo = bits[offset:offset + self.preamble.size]
        my_echo = bits[offset + self.preamble.size:offset + 2 * self.preamble.size]
        return spy, hdr, new_echo, my_echo

    def assemble_packet(self, new_echo, valid):
        spy = self.preamble[:self.spy_length]
        hdr = bytes_to_bits(self.reedsolomon.encode(struct.pack('BH', valid, self.train_cnt * 2)))
        # Return classic mod section, neural mod section
        return np.concatenate([spy, hdr]), np.concatenate([self.preamble, new_echo])

    def update_version(self):
        """
        Increment the version counter
        """
        self.version += 1
        return self.version

