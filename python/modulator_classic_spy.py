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

import struct
import time
import uuid

from gnuradio import gr
import numpy as np
import pmt
import reedsolo

from torch_echo.modulators import ModulatorClassic
from torch_echo.utils import util_data


def bytes_to_bits(input):
    return np.unpackbits(np.array(input)).astype(np.bool)


def bits_to_bytes(input):
    return bytearray(np.packbits(input))


class modulator_classic_spy(gr.basic_block, ModulatorClassic):
    """
    Initialize parameters used for modulation
    Inputs:
    bits_per_symbol: Number of bits per symbol
    preamble: pseudo-random sequence used to update the modulator after a round trip
    log_ber_interval: number of updates between logging the round trip BER
    spy_length: number of bits used to 'spy' on packets to ensure that they aren't inordinately corrupted
    """
    def __init__(self, bits_per_symbol, preamble=None, log_ber_interval=10, 
                 spy_length=64, spy_threshold=0.1):
        ModulatorClassic.__init__(self, bits_per_symbol, max_amplitude=0.09)
        gr.basic_block.__init__(self,
                                name="modulator_classic",
                                in_sig=None,
                                out_sig=None)
        # Echo protocol variables
        assert preamble is not None, "Preamble must be provided"
        if preamble is not np.ndarray:
            preamble = np.array(preamble)
        self.preamble = preamble
        self.preamble_si = util_data.bits_to_integers(self.preamble, self.bits_per_symbol)
        self.log_ber_interval = log_ber_interval

        # Message port setup and variables
        self.port_id_in = pmt.intern("bits")
        self.port_id_update = pmt.intern("update")
        self.port_id_out = pmt.intern("symbols")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_in(self.port_id_update)
        self.message_port_register_out(self.port_id_out)
        self.set_msg_handler(self.port_id_in, self.handle_packet)
        self.set_msg_handler(self.port_id_update, self.handle_update)
        self.packet_cnt = 0
        self.ber_cnt = 0

        # Packet header and spy variables
        self.spy_length = spy_length
        assert self.spy_length % self.bits_per_symbol == 0
        self.spy_threshold = spy_threshold
        self.reedsolomon = reedsolo.RSCodec(4)
        self.rs_length = 4 * 2 * 8  # 4 bytes data, 4 bytes parity, 8 bits per byte

        # Logging stuff
        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("classic mod {}: {} bits per symbol".format(self.uuid_str, self.bits_per_symbol))
        with open("ber_echo_{}.csv".format(self.uuid_str), "w") as f:
            f.write("train_iter,BER\n")

    def handle_packet(self, pdu):
        t0 = time.time()
        self.packet_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.to_python(pmt.cdr(pdu))
        bits = self.assemble_packet(vec[self.preamble.size:], valid=False)
        data_si = util_data.bits_to_integers(bits, self.bits_per_symbol)
        symbs = self.modulate(data_si).astype(np.complex64)
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs)))
        t1 = time.time()
        self.logger.debug("classic mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def handle_update(self, pdu):
        t0 = time.time()
        self.packet_cnt += 1
        self.ber_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.to_python(pmt.cdr(pdu))
        spy, hdr, new_echo, my_echo = self.split_packet(vec)
        spy_ber = sum(spy != self.preamble[:self.spy_length]) * 1.0 / self.spy_length
        if hdr is not None:
            valid = hdr[0]
            pktidx = hdr[1]
        else:
            valid = False
        if (self.ber_cnt % self.log_ber_interval == 0 and
                valid and
                spy_ber < self.spy_threshold):
            ###DEBUG###
            np.save("clmod_preamble_{}".format(pktidx), new_echo)
            np.save("clmod_echo_{}".format(pktidx), my_echo)
            #np.save("clmod_preamble_{}".format(self.ber_cnt * 2 + 1), new_echo)
            #np.save("clmod_echo_{}".format(self.ber_cnt * 2 + 1), my_echo)
            ###DEBUG###
            ber = sum(my_echo != self.preamble) * 1.0 / self.preamble.size
            with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                f.write("{},{}\n".format(self.ber_cnt, ber))
        bits = self.assemble_packet(new_echo, valid=spy_ber < self.spy_threshold)
        data_si = util_data.bits_to_integers(bits, self.bits_per_symbol)
        symbs = self.modulate(data_si).astype(np.complex64)
        ###DEBUG###
        if (self.ber_cnt % self.log_ber_interval == 0 and
                valid and
                spy_ber < self.spy_threshold):
            np.save("clmod_symbs_{}".format(self.ber_cnt * 2 + 1), 
                    symbs[-2 * self.preamble_si.size:])
        ###DEBUG###
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs)))
        t1 = time.time()
        self.logger.debug("classic mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, bits.size, t1 - t0))

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
        hdr = bytes_to_bits(self.reedsolomon.encode(struct.pack('BH', valid, self.ber_cnt * 2 + 1)))
        # Return classic mod section, neural mod section
        return np.concatenate([spy, hdr, self.preamble, new_echo])

