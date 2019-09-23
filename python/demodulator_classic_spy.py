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

from torch_echo.demodulators import DemodulatorClassic
from torch_echo.utils import util_modulation, util_data


def bytes_to_bits(input):
    return np.unpackbits(np.array(input)).astype(np.bool)


def bits_to_bytes(input):
    return bytearray(np.packbits(input))


class demodulator_classic_spy(gr.basic_block, DemodulatorClassic):
    """
    Initialize parameters used for demodulation
    Inputs:
    bits_per_symbol: bits per symbol, determines modulation scheme
    block_length: block length to break demodulated input into to avoid excessive memory usage
    preamble: pseudo-random sequence used to update the demodulator and determine BER
    log_ber_interval: number of updates between logging the BER
    spy_length: number of bits used to 'spy' on packets to ensure that they aren't inordinately corrupted
    spy_threshold: fraction of corrupted bits required for a packet to be 'corrupted'
                   make sure this is well above the expected error rate!
    """

    def __init__(self, bits_per_symbol, block_length=10000, preamble=None, log_ber_interval=10,
                 spy_length=64, spy_threshold=0.1, alias="demod_classic"):
        DemodulatorClassic.__init__(self, bits_per_symbol=bits_per_symbol, block_length=block_length, max_amplitude=0.09)
        gr.basic_block.__init__(self,
                                name="demod_classic",
                                in_sig=None,
                                out_sig=None)
        self.alias = alias
        self.port_id_in = pmt.intern("symbols")
        self.port_id_out = pmt.intern("bits")
        self.port_id_corrupt = pmt.intern("corrupt")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_out(self.port_id_out)
        self.message_port_register_out(self.port_id_corrupt)
        self.set_msg_handler(self.port_id_in, self.handle_packet)
        if preamble is not np.ndarray:
            preamble = np.array(preamble)
        self.preamble = preamble
        self.packet_cnt = 0
        self.update_cnt = 0
        self.ber = None
        self.log_ber_interval = log_ber_interval

        self.spy_length = spy_length
        assert self.spy_length % self.bits_per_symbol == 0
        self.spy_threshold = spy_threshold
        self.reedsolomon = reedsolo.RSCodec(4)
        self.rs_length = 4 * 2 * 8  # 4 bytes data, 4 bytes parity, 8 bits per byte

        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("classic demod {}: {} bits per symbol".format(self.uuid_str, self.bits_per_symbol))
        with open("ber_{}.csv".format(self.uuid_str), "w") as f:
            f.write("iter,BER\n")

    def handle_packet(self, pdu):
        t0 = time.time()
        self.packet_cnt += 1
        tag_dict = pmt.car(pdu)
        vec = pmt.to_python(pmt.cdr(pdu))
        _, _, new_echo_s, my_echo_s = self.split_packet_iq(vec)
        ###DEBUG###
        if self.alias == "classic-agent":
            n = self.spy_length / 2
            vec[n:] += (np.random.randn(vec.size-n) + 1j * np.random.randn(vec.size-n)) * 0.1
        ###DEBUG###
        bits = util_data.integers_to_bits(self.demodulate(vec), self.bits_per_symbol)
        spy, hdr, new_echo, _ = self.split_packet_bits(bits)
        if hdr is not None:
            valid = hdr[0]
            pktidx = hdr[1]
        else:
            valid = False
        # Check spy header to see if packet is corrupt
        if self.spy_length > 0:
            spy_ber = sum(spy != self.preamble[:self.spy_length]) * 1.0 / self.spy_length
        else:
            spy_ber = 0
        if spy_ber > self.spy_threshold:
            # BAD PACKET!
            self.logger.debug("classic demod {} spy ber {} above threshold {}".format(
                              self.uuid_str, spy_ber, self.spy_threshold))
            # Publish to both ports so mod can decide what to do with the bad packet
            self.message_port_pub(self.port_id_corrupt,
                                  pmt.cons(pmt.PMT_NIL, pmt.to_pmt(bits.astype(np.int8))))
            self.message_port_pub(self.port_id_out,
                                  pmt.cons(pmt.PMT_NIL, pmt.to_pmt(bits.astype(np.int8))))
        else:
            # Publish good packet, without spy header
            self.update_cnt += 1
            ###DEBUG###
            if self.alias == "neural-agent":
                np.save("received_preamble_{}".format(pktidx), new_echo_s)
                np.save("received_echo_{}".format(pktidx), my_echo_s)
            else:
                np.save("received_classic_preamble_{}".format(pktidx), new_echo_s)
                np.save("received_classic_echo_{}".format(pktidx), my_echo_s)
            ###DEBUG###
            self.message_port_pub(self.port_id_out,
                                  pmt.cons(pmt.PMT_NIL,
                                           pmt.to_pmt(bits.astype(np.int8))))

            self.ber = sum(self.preamble != new_echo) * 1.0 / self.preamble.size
            if self.update_cnt % self.log_ber_interval == 0:
                with open("ber_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(self.update_cnt, self.ber))
        t1 = time.time()
        self.logger.debug("classic demod {} handled {} bits in {} seconds".format(
                          self.uuid_str, bits.size, t1 - t0))
    
    def split_packet_iq(self, iq):
        idx = self.spy_length / self.bits_per_symbol
        spy = iq[:idx]
        hdr = iq[idx:idx + self.rs_length / self.bits_per_symbol]
        idx += self.rs_length / self.bits_per_symbol
        new_echo = iq[idx:idx + self.preamble.size / self.bits_per_symbol]
        idx += self.preamble.size / self.bits_per_symbol
        my_echo = iq[idx:idx + self.preamble.size / self.bits_per_symbol]
        return spy, hdr, new_echo, my_echo

    def split_packet_bits(self, bits):
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

