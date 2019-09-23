#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Josh Sanz <jsanz@berkeley.edu>
# 2019 09 13
#
# Copyright 2019 <+YOU OR YOUR COMPANY+>.
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

import struct
import time
import uuid

from gnuradio import gr
import numpy as np
import pmt
import reedsolo
import torch

from torch_echo.modulators import ModulatorClassic, ModulatorNeural
from torch_echo.demodulators import DemodulatorClassic, DemodulatorNeural
from torch_echo.utils import util_data
from torch_echo.utils.visualize import gen_demod_grid


def bytes_to_bits(input):
    return np.unpackbits(np.array(input)).astype(np.bool)


def bits_to_bytes(input):
    return bytearray(np.packbits(input))


class echo_mod_demod(gr.basic_block):
    MODTYPES = ('classic', 'neural')
    """
    docstring for block echo_mod_demod
    """
    def __init__(self, npreamble, shared_preamble, bits_per_symb, modtype, demodtype,
                 mod_seed=128, demod_seed=256,
                 mod_hidden_layers=[64], demod_hidden_layers=[64],
                 mod_init_weights="", demod_init_weights="",
                 log_interval=20, spy_length=64, spy_threshold=0.1, 
                 max_amplitude=0., lambda_center=0.1,
                 _alias=""):
        assert modtype in echo_mod_demod.MODTYPES, "modtype must be one of {}".format(echo_mod_demod.MODTYPES)
        assert demodtype in echo_mod_demod.MODTYPES, "demodtype must be one of {}".format(echo_mod_demod.MODTYPES)
        gr.basic_block.__init__(self,
            name="echo-{}-mod-{}-demod".format(modtype, demodtype),
            in_sig=None,
            out_sig=None)
        # Debug variables
        self.alias = _alias

        # Echo protocol variables
        self.bits_per_symbol = bits_per_symb
        self.npreamble = npreamble
        # None or empty array ==> private rolling preambles
        self.use_shared = (shared_preamble is not None and
                           len(shared_preamble) > 0)
        if self.use_shared and not isinstance(shared_preamble, np.ndarray):
            self.preamble = np.load(shared_preamble)[:npreamble]
        elif self.use_shared:
            self.preamble = shared_preamble
        else:
            self.preamble = np.random.randint(0, 2, self.npreamble)
        self.preamble_si = util_data.bits_to_integers(self.preamble, bits_per_symb)
        np.save("preamble_si_0", self.preamble_si)
        ###DEBUG###
        self.preamble_hist = {0: self.preamble}
        self.actions_hist = {}
        ###DEBUG###

        self.modtype = modtype
        self.demodtype = demodtype
        self.run_mode = "train"
        if modtype == 'classic':
            self.mod = ModulatorClassic(self.bits_per_symbol, max_amplitude=max_amplitude)
        else:
            self.mod = ModulatorNeural(
                    seed=mod_seed,
                    hidden_layers=list(mod_hidden_layers),
                    bits_per_symbol=bits_per_symb,
                    lampda_p=0.0,
                    max_std=100,
                    min_std=1e-1,
                    initial_std=2.0e-1,
                    restrict_energy=1,
                    activation_fn_hidden='tanh',
                    activation_fn_output=None,
                    optimizer='adam',
                    lambda_prob=1e-10,
                    stepsize_mu=1e-3,
                    stepsize_sigma=1e-4,
                    max_amplitude=max_amplitude,
                    lambda_center=lambda_center)
            # manage model weights
            if len(mod_init_weights) > 0:
                self.mod.model.load_state_dict(torch.load(mod_init_weights))
                torch.save(self.mod.model.state_dict(), "./mod-init-weights.mdl")
                self.run_mode = "freeze"
            # create spy modulator
            self.spy_mod = ModulatorClassic(self.bits_per_symbol, max_amplitude=max_amplitude)
            self.past_actions = None
        if demodtype == 'classic':
            self.demod = DemodulatorClassic(self.bits_per_symbol, max_amplitude=max_amplitude)
        else:
            self.demod = DemodulatorNeural(
                    seed=demod_seed,
                    hidden_layers=list(demod_hidden_layers),
                    bits_per_symbol=bits_per_symb,
                    activation_fn_hidden='tanh',
                    activation_fn_output=None,
                    optimizer='adam',
                    loss_type='l2',
                    stepsize_cross_entropy=1e-2,
                    cross_entropy_weight=1.0)
            if len(demod_init_weights) > 0:
                self.demod.model.load_state_dict(torch.load(demod_init_weights))
                torch.save(self.demod.model.state_dict(), "./demod-init-weights.mdl")
                self.run_mode = "freeze"
            self.spy_demod = DemodulatorClassic(self.bits_per_symbol, max_amplitude=max_amplitude)

        # Message port setup and variables
        self.port_id_mod_in = pmt.intern("mod_in")
        self.port_id_mod_out = pmt.intern("mod_out")
        self.port_id_demod_in = pmt.intern("demod_in")
        self.port_id_demod_out = pmt.intern("demod_out")
        self.port_id_ctrl = pmt.intern("control")
        self.message_port_register_in(self.port_id_mod_in)
        self.message_port_register_in(self.port_id_demod_in)
        self.message_port_register_in(self.port_id_ctrl)
        self.message_port_register_out(self.port_id_mod_out)
        self.message_port_register_out(self.port_id_demod_out)
        self.set_msg_handler(self.port_id_mod_in, self.handle_mod)
        self.set_msg_handler(self.port_id_demod_in, self.handle_demod)
        self.set_msg_handler(self.port_id_ctrl, self.handle_control)
        self.mod_packet_cnt = 0
        self.demod_packet_cnt = 0
        self.mod_update_cnt = 0
        self.demod_update_cnt = 0

        # Packet header and spy variables
        self.reedsolomon = reedsolo.RSCodec(6)
        self.rs_length = 6 * 2 * 8  # 6 bytes data, 6 bytes parity, 8 bits per byte
        self.spy_length = spy_length
        assert self.spy_length % self.bits_per_symbol == 0
        self.spy_threshold = spy_threshold
        if self.use_shared:
            self.spy_master = self.preamble[:spy_length]
            assert self.spy_master.size >= spy_length, "shared preamble did not contain sufficient data to fill the spy field"
        else:
            try:
                self.spy_master = np.load("spy_master.npy")[:self.spy_length]
            except IOError as e:
                print("If using private preambles, you must provide a spy_master.npy file containing the spy field bits")
                raise e
            assert self.spy_master.size >= spy_length, "spy_master.npy did not contain sufficient data to fill the spy field"

        # Logging stuff
        self.log_interval = log_interval
        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("mod-demod {}: {} bits per symbol".format(self.uuid_str, self.bits_per_symbol))
        with open("ber_{}.csv".format(self.uuid_str), "w") as f:
            f.write("train_iter,BER\n")
        with open("ber_echo_{}.csv".format(self.uuid_str), "w") as f:
            f.write("train_iter,BER\n")
        if len(mod_init_weights) > 0:
            self.logger.info("neural mod {}: initialized weights from {}".format(self.uuid_str, mod_init_weights))
        if len(demod_init_weights) > 0:
            self.logger.info("neural mod {}: initialized weights from {}".format(self.uuid_str, demod_init_weights))

    def handle_mod(self, pdu):
        if self.modtype == 'classic':
            return self.handle_mod_classic(pdu)
        else:
            return self.handle_mod_neural(pdu)

    def handle_demod(self, pdu):
        if self.demodtype == 'classic':
            rval = self.handle_demod_classic(pdu)
        else:
            rval = self.handle_demod_neural(pdu)
        if self.modtype == 'classic':
            return self.handle_mod_feedback_classic(rval)
        else:
            return self.handle_mod_feedback_neural(rval)

    ################ CLASSIC #################
    def handle_mod_classic(self, pdu):
        t0 = time.time()
        self.mod_packet_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.to_python(pmt.cdr(pdu))
        if not self.use_shared:
            self.gen_preamble()
        bits = self.assemble_packet(vec[self.preamble.size:], (1 << 16) - 1, valid=False)
        data_si = util_data.bits_to_integers(bits, self.bits_per_symbol)
        symbs = self.mod.modulate(data_si).astype(np.complex64)
        self.message_port_pub(self.port_id_mod_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs)))
        t1 = time.time()
        self.logger.debug("classic mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, bits.size, t1 - t0))

    def handle_mod_feedback_classic(self, vec):
        t0 = time.time()
        self.mod_packet_cnt += 1
        self.mod_update_cnt += 1
        spy, hdr, new_echo, my_echo = self.split_packet_bits(vec)
        spy_ber = sum(spy != self.spy_master) * 1.0 / self.spy_length
        if hdr is not None:
            idxpre = hdr[0]
            idxecho = hdr[1]
            valid = hdr[2]
            if idxecho != self.mod_update_cnt - 1:
                print("Update cnt {} echo idx {}".format(self.mod_update_cnt, idxecho))
        else:
            valid = False
            idxpre = (1 << 16) - 1
            idxecho = (1 << 16) - 1
        if ((self.mod_update_cnt % self.log_interval == 0 or self.run_mode == 'freeze') and
                valid and
                spy_ber < self.spy_threshold):
            try:
                preamble = self.get_preamble_hist(idxecho, pop=True)
                print("{} classic offset {}".format(self.uuid_str, self.mod_update_cnt - idxecho))
                ber = sum(my_echo != preamble) * 1.0 / self.preamble.size
                with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(self.mod_update_cnt, ber))
            except KeyError as e:
                self.logger.info("DEBUG::Unable to calculate BER with stored index {}".format(idxecho))
        if not self.use_shared:
            self.gen_preamble()
        bits = self.assemble_packet(new_echo, idxpre, valid=spy_ber < self.spy_threshold)
        data_si = util_data.bits_to_integers(bits, self.bits_per_symbol)
        symbs = self.mod.modulate(data_si).astype(np.complex64)
        if self.mod_update_cnt - idxecho == 1 or idxecho == 65535:
            self.message_port_pub(self.port_id_mod_out,
                                  pmt.cons(pmt.PMT_NIL,
                                           pmt.to_pmt(symbs)))
        else:
            print("Not sending new pkt because {} or {}".format(
                    self.mod_update_cnt - idxecho, idxecho != 65535))
        t1 = time.time()
        self.logger.debug("classic mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, bits.size, t1 - t0))

    def handle_demod_classic(self, pdu):
        t0 = time.time()
        self.demod_packet_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.to_python(pmt.cdr(pdu))
        _, _, new_echo_s, my_echo_s = self.split_packet_iq(vec)
        bits = util_data.integers_to_bits(self.demod.demodulate(vec), self.bits_per_symbol)
        spy, hdr, new_echo, _ = self.split_packet_bits(bits)
        # Check spy header to see if packet is corrupt
        if self.spy_length > 0:
            spy_ber = sum(spy != self.spy_master) * 1.0 / self.spy_length
        else:
            spy_ber = 0
        # Interpret header
        if hdr is not None:
            idxpre = hdr[0]
            idxecho = hdr[1]
            valid = hdr[2]
        else:
            valid = False
            idxpre = (1 << 16) - 1
            idxecho = (1 << 16) - 1
            if spy_ber < self.spy_threshold:
                self.logger.debug("classic demod {} spy passed ({}) but header failed to decode".format(self.uuid_str, spy_ber))
        if spy_ber > self.spy_threshold:
            # BAD PACKET!
            self.logger.debug("classic demod {} spy ber {} above threshold {}".format(
                              self.uuid_str, spy_ber, self.spy_threshold))
            # Publish to both ports so mod can decide what to do with the bad packet
            #self.message_port_pub(self.port_id_corrupt,
            #                      pmt.cons(pmt.PMT_NIL, pmt.to_pmt(bits.astype(np.int8))))
            self.message_port_pub(self.port_id_demod_out,
                                  pmt.cons(pmt.PMT_NIL, pmt.to_pmt(bits.astype(np.int8))))
        else:
            # Publish good packet
            self.demod_update_cnt += 1
            self.message_port_pub(self.port_id_demod_out,
                                  pmt.cons(pmt.PMT_NIL,
                                           pmt.to_pmt(bits.astype(np.int8))))

            try:
                preamble = self.get_preamble_hist(idxecho)[:self.preamble.size]
                ber = sum(preamble != new_echo) * 1.0 / self.preamble.size
                if self.demod_update_cnt % self.log_interval == 0 or self.run_mode == 'freeze':
                    with open("ber_{}.csv".format(self.uuid_str), "a") as f:
                        f.write("{},{}\n".format(self.demod_update_cnt, ber))
            except KeyError as e:
                self.logger.info("DEBUG::Unable to calculate BER with stored index {}".format(idxecho))
        t1 = time.time()
        self.logger.debug("classic demod {} handled {} bits in {} seconds".format(
                          self.uuid_str, bits.size, t1 - t0))
        return bits

    ################ NEURAL #################
    def handle_mod_neural(self, pdu):
        t0 = time.time()
        self.mod_packet_cnt += 1
        tag_dict = pmt.to_python(pmt.car(pdu))
        vec = pmt.to_python(pmt.cdr(pdu))
        exploration = 'exploit' if self.run_mode == 'freeze' else 'explore'
        symbs = self.mod.modulate(util_data.bits_to_integers(vec, self.bits_per_symbol), exploration)
        self.past_actions = symbs
        ###DEBUG###
        self.actions_hist[self.mod_packet_cnt] = symbs
        ###DEBUG###
        # Spy and header fields
        if not self.use_shared:
            self.gen_preamble()
        classic, _ = self.assemble_packet_neural(vec[self.preamble.size:], (1 << 16) - 1, valid=False)
        classic_si = util_data.bits_to_integers(classic, self.bits_per_symbol)
        symbs = np.concatenate([self.spy_mod.modulate(classic_si), symbs])
        self.message_port_pub(self.port_id_mod_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs.astype(np.complex64))))
        t1 = time.time()
        self.logger.debug("neural mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def handle_mod_feedback_neural(self, vec):
        t0 = time.time()
        self.mod_packet_cnt += 1
        self.mod_update_cnt += 1
        # Split packet into functional parts
        spy, hdr, new_echo, my_echo = self.split_packet_bits(vec)
        spy_ber = sum(spy != self.spy_master) * 1.0 / self.spy_length
        if hdr is not None:
            idxpre = hdr[0]
            idxecho = hdr[1]
            valid = hdr[2]
        else:
            valid = False
            idxpre = (1 << 16) - 1
            idxecho = (1 << 16) - 1
        # Do not perform updates while frozen
        if self.run_mode == "freeze" and spy_ber < self.spy_threshold and valid:
            try:
                # Save every round trip BER while frozen to measure average performance
                preamble = self.get_preamble_hist(idxecho, pop=True)
                print("{} neural offset {}".format(self.uuid_str, self.mod_update_cnt - idxecho))
                ber = sum(np.abs(preamble - my_echo)) * 1.0 / self.preamble.size
                with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(self.mod_update_cnt, ber))
            except KeyError as e:
                self.logger.info("Unable to calculate BER with stored index {}".format(idxecho))
        elif spy_ber < self.spy_threshold and valid:  # run_mode == "train"
            try:
                labels_si_g = util_data.bits_to_integers(my_echo, self.bits_per_symbol)
                labels_all = util_data.bits_to_integers(np.concatenate([new_echo, my_echo]),
                                              self.bits_per_symbol)
                #p_actions = self.past_actions[0:self.preamble_si.shape[0]]  # actions
                ###DEBUG###
                p_actions = self.get_actions_hist(idxecho, pop=True)[:self.preamble_si.size]
                print("{} neural offset {}".format(self.uuid_str, self.mod_update_cnt - idxecho))
                preamble = self.get_preamble_hist(idxecho, pop=True)
                ###DEBUG###

                # Only use preamble from feedback, at least for now
                preamble_si = util_data.bits_to_integers(preamble, self.bits_per_symbol)
                reward, _, _, loss = self.mod.update(preamble_si=preamble_si,  # labels
                                                     actions=p_actions,
                                                     labels_si_g=labels_si_g)  # demodulator guesses
                self.version_dirty = False
                if self.mod_update_cnt % self.log_interval == 0:
                    print("Saving mod constellation @{}".format(self.mod_update_cnt))
                    np.save("neural_mod_constellation_{:05d}_{}".format(self.mod_update_cnt,
                               time.strftime('%Y%m%d_%H%M%S')),
                               self.mod.get_constellation())
                    ber = sum(preamble != my_echo) * 1.0 / preamble.size
                    with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                        f.write("{},{}\n".format(self.mod_update_cnt, ber))
            except KeyError as e:
                self.logger.info("Unable to train modulator with stored index {}".format(idxecho))
        # Send the echo with a new preamble to continue the training cycle
        if not self.use_shared:
            self.gen_preamble()
        classic, neural = self.assemble_packet_neural(new_echo, idxpre, valid=spy_ber < self.spy_threshold)

        cl_si = util_data.bits_to_integers(classic, self.bits_per_symbol)
        cl_symb = self.spy_mod.modulate(cl_si)

        nn_si = util_data.bits_to_integers(neural, self.bits_per_symbol)
        exploration = 'exploit' if self.run_mode == 'freeze' else 'explore'
        nn_symb = self.mod.modulate(nn_si, exploration)
        self.past_actions = nn_symb
        ###DEBUG###
        self.actions_hist[self.mod_update_cnt] = nn_symb
        ###DEBUG###

        packet = np.concatenate([cl_symb, nn_symb])
        if self.mod_update_cnt - idxecho == 1 or idxecho == 65535:
            self.message_port_pub(self.port_id_mod_out,
                                  pmt.cons(pmt.PMT_NIL,
                                           pmt.to_pmt(packet.astype(np.complex64))))
        else:
            print("Not sending new pkt because {} or {}".format(
                    self.mod_update_cnt - idxecho, idxecho != 65535))
        # Logging
        t1 = time.time()
        self.logger.debug("neural mod {} updated with {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def handle_demod_neural(self, pdu):
        t0 = time.time()
        self.demod_packet_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.cdr(pdu)
        # tag_p = pmt.to_python(tag)
        data_c = pmt.to_python(vec)
        nbits = data_c.size * self.bits_per_symbol
        spy_s, hdr_s, new_echo_s, my_echo_s = self.split_packet_iq(data_c)
        spy_b = util_data.integers_to_bits(self.spy_demod.demodulate(spy_s), self.bits_per_symbol)
        hdr_b = util_data.integers_to_bits(self.spy_demod.demodulate(hdr_s), self.bits_per_symbol)
        exploration = 'exploit' if self.run_mode == 'freeze' else 'explore'
        new_echo_b = util_data.integers_to_bits(self.demod.demodulate(
                new_echo_s, mode=exploration), self.bits_per_symbol)
        my_echo_b = util_data.integers_to_bits(self.demod.demodulate(
                my_echo_s, mode=exploration), self.bits_per_symbol)
        bits = np.concatenate([spy_b, hdr_b, new_echo_b, my_echo_b])
        _, hdr, _, _ = self.split_packet_bits(bits)
        if hdr is not None:
            idxpre = hdr[0]
            idxecho = hdr[1]
            valid = hdr[2]
        else:
            valid = False
        # Check spy field for corruption
        if self.spy_length > 0:
            spy_ber = sum(spy_b != self.spy_master) * 1.0 / self.spy_length
        else:
            spy_ber = 0
        if spy_ber > self.spy_threshold:
            # BAD PACKET!
            self.logger.debug("neural demod {} spy ber {} above threshold {}".format(
                              self.uuid_str, spy_ber, self.spy_threshold))
            # Publish to both ports so mods can decide how to handle corruptions
            self.message_port_pub(self.port_id_demod_out,
                                  pmt.cons(pmt.PMT_NIL, pmt.to_pmt(bits.astype(np.int8))))
        elif valid:
            # Good packet, demod and train with the neural agent
            self.demod_update_cnt += 1
            # update
            if self.run_mode == "train" and self.use_shared:
                self.demod_train_preamble(new_echo_s, idxpre)
            elif self.run_mode == "train" and not self.use_shared:
                self.demod_train_preamble(my_echo_s, idxecho)
            # demod
            #new_echo_b = util_data.integers_to_bits(self.demod.demodulate(new_echo_s, 'exploit'), self.bits_per_symbol)
            #my_echo_b = util_data.integers_to_bits(self.demod.demodulate(my_echo_s, 'exploit'), self.bits_per_symbol)
            #labels_g = np.concatenate([spy_b, hdr_b, new_echo_b, my_echo_b])
            labels_g = bits  # Match python simulation behavior
            # Sample some BERs during training, but sample all while frozen to collect average
            # behavior faster
            if self.demod_update_cnt % self.log_interval == 0 or self.run_mode == "freeze":
                ber = sum(new_echo_b != self.preamble) * 1.0 / self.preamble.size
                with open("ber_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(self.demod_packet_cnt, ber))
            # publish
            self.message_port_pub(self.port_id_demod_out,
                                  pmt.cons(pmt.PMT_NIL,
                                           pmt.to_pmt(labels_g.astype(dtype=np.int8))))
        t1 = time.time()
        self.logger.debug("neural demod {} handled {} bits in {} seconds".format(
                          self.uuid_str, nbits, t1 - t0))
        return bits

    def demod_train_preamble(self, data_c, idxecho):
        try:
            preamble_si = util_data.bits_to_integers(self.get_preamble_hist(idxecho),
                                                     self.bits_per_symbol)
            self.demod.update(inputs=preamble_si, actions=[],
                              data_for_rewards=data_c, mode='echo')
            if self.demod_packet_cnt % self.log_interval == 0:
                print("Saving demod constellation @{}".format(self.demod_packet_cnt))
                data_vis = gen_demod_grid(points_per_dim=100, min_val=-2.5, max_val=2.5)['data']
                labels_si_g = self.demod.demodulate(data_c=data_vis, mode='exploit')
                np.savez("neural_demod_constellation_{:05d}_{}".format(self.demod_packet_cnt,
                            time.strftime('%Y%m%d_%H%M%S')),
                            iq=data_vis, labels=labels_si_g)
        except KeyError as e:
            self.logger.info("Unable to train demodulator with stored index {}".format(idxecho))

    ################ UTILS #################
    def gen_preamble(self):
        self.preamble = np.random.randint(0, 2, self.npreamble)
        self.preamble_si = util_data.bits_to_integers(self.preamble, self.bits_per_symbol)
        np.save("preamble_si_{}".format(self.mod_packet_cnt), self.preamble_si)
        ###DEBUG###
        self.preamble_hist[self.mod_update_cnt] = self.preamble
        ###DEBUG###

    def split_packet_bits(self, bits):
        spy = bits[:self.spy_length]
        try:
            hdr = struct.unpack('HHH',
                    self.reedsolomon.decode(bits_to_bytes(
                        bits[self.spy_length:self.spy_length + self.rs_length])))
        except reedsolo.ReedSolomonError as e:
            hdr = None
        offset = self.spy_length + self.rs_length
        new_echo = bits[offset:offset + self.preamble.size]
        my_echo = bits[offset + self.preamble.size:offset + 2 * self.preamble.size]
        return spy, hdr, new_echo, my_echo

    def split_packet_iq(self, iq):
        idx = self.spy_length / self.bits_per_symbol
        spy = iq[:idx]
        hdr = iq[idx:idx + self.rs_length / self.bits_per_symbol]
        idx += self.rs_length / self.bits_per_symbol
        new_echo = iq[idx:idx + self.preamble.size / self.bits_per_symbol]
        idx += self.preamble.size / self.bits_per_symbol
        my_echo = iq[idx:idx + self.preamble.size / self.bits_per_symbol]
        return spy, hdr, new_echo, my_echo

    def assemble_packet(self, echo, echo_idx, valid):
        return np.concatenate(self.assemble_packet_neural(echo, echo_idx, valid))

    def assemble_packet_neural(self, echo, echo_idx, valid):
        # We don't need a short for the valid flag, but it helps keep the encoded
        # header a multiple of 3 bits so 8PSK doesn't break
        hdr = bytes_to_bits(self.reedsolomon.encode(
                            struct.pack('HHH', self.mod_update_cnt, echo_idx, valid)))
        # Return classic mod section, neural mod section
        return np.concatenate([self.spy_master, hdr]), np.concatenate([self.preamble, echo])

    def get_preamble_hist(self, idx, pop=False):
        if self.use_shared:
            return self.preamble
        if pop:
            preamble = self.preamble_hist.pop(idx)
        else:
            preamble = self.preamble_hist[idx]
        return preamble

    def get_actions_hist(self, idx, pop=False):
        if pop:
            actions = self.actions_hist.pop(idx)
        else:
            actions = self.actions_hist[idx]
        return actions

    ################ ZMQ CONTROL #################
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
        self.logger.debug("echo agent received command with keys: {}".format(keys))
        self.handle_control_mod(keys)
        self.handle_control_demod(keys)

    def handle_control_mod(self, keys):
        # Check for keys we care about
        if "freeze" in keys:
            self.run_mode = "freeze"
            if self.modtype == 'neural':
                self.logger.info("neural mod {}: freezing model and saving state".format(self.uuid_str))
                torch.save(self.mod.model.state_dict(), "mod_neural_weights-{}.mdl".format(self.uuid_str))
                print("Saving final mod constellation @{}".format(self.mod_update_cnt))
                np.save("neural_mod_constellation_{:05d}_{}".format(self.mod_update_cnt,
                           time.strftime('%Y%m%d_%H%M%S')),
                           self.mod.get_constellation())
        elif "train" in keys:
            self.logger.info("neural mod {}: resuming training".format(self.uuid_str))
            self.run_mode = "train"

    def handle_control_demod(self, keys):
        # Check for keys we care about
        if "freeze" in keys:
            self.run_mode = "freeze"
            if self.demodtype == 'neural':
                self.logger.info("neural demod {}: freezing model and saving state".format(self.uuid_str))
                torch.save(self.demod.model.state_dict(), "demod_neural_weights-{}.mdl".format(self.uuid_str))
                print("Saving final demod constellation @{}".format(self.demod_update_cnt))
                data_vis = gen_demod_grid(points_per_dim=100, min_val=-2.5, max_val=2.5)['data']
                labels_si_g = self.demod.demodulate(data_c=data_vis, mode='exploit')
                np.savez("neural_demod_constellation_{:05d}_{}".format(self.demod_update_cnt,
                            time.strftime('%Y%m%d_%H%M%S')),
                            iq=data_vis, labels=labels_si_g)
        elif "train" in keys:
            self.run_mode = "train"
            self.logger.info("neural demod {}: resuming training".format(self.uuid_str))

