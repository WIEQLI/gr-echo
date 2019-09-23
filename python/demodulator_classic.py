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
import pmt
from gnuradio import gr
from torch_echo.demodulators import DemodulatorClassic
from torch_echo.utils import util_modulation, util_data


class demodulator_classic(gr.basic_block, DemodulatorClassic):
    """
    Initialize parameters used for demodulation
    Inputs:
    bits_per_symbol: bits per symbol, determines modulation scheme
    preamble: pseudo-random sequence used to update the demodulator and update the BER
    log_ber_interval: number of updates between logging the BER
    block_length: block length to break demodulated input into to avoid excessive memory usage
    """

    def __init__(self, bits_per_symbol, block_length=10000, preamble=None, log_ber_interval=10):
        DemodulatorClassic.__init__(self, bits_per_symbol=bits_per_symbol, block_length=block_length)
        gr.basic_block.__init__(self,
                                name="demod_classic",
                                in_sig=None,
                                out_sig=None)
        self.port_id_in = pmt.intern("symbols")
        self.port_id_out = pmt.intern("bits")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_out(self.port_id_out)
        self.set_msg_handler(self.port_id_in, self.handle_packet)
        if preamble is not numpy.ndarray:
            preamble = numpy.array(preamble)
        self.preamble = preamble
        self.packet_cnt = 0
        self.ber = None
        self.log_ber_interval = log_ber_interval

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
        bits = util_data.integers_to_bits(self.demodulate(vec), self.bits_per_symbol)
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(bits.astype(numpy.int8))))
        if self.preamble is not None:
            # Compare decoded preamble to known version to get a BER estimate
            errs = 0
            for a, b in zip(self.preamble, bits):
                if a != b:
                    errs += 1
            self.ber = errs * 1.0 / len(self.preamble)
            if self.packet_cnt % self.log_ber_interval == 0:
                with open("ber_{}.csv".format(self.uuid_str), "a") as f:
                    f.write("{},{}\n".format(self.packet_cnt, self.ber))
        t1 = time.time()
        self.logger.debug("classic demod {} handled {} bits in {} seconds".format(
                          self.uuid_str, bits.size, t1 - t0))
