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
from torch_echo.modulators import ModulatorClassic
from torch_echo.utils import util_data


class modulator_classic(gr.basic_block, ModulatorClassic):
    """
    Initialize parameters used for modulation
    Inputs:
    bits_per_symbol: Number of bits per symbol
    preamble: pseudo-random sequence used to update the modulator after a round trip
    log_ber_interval: number of updates between logging the round trip BER
    modulation_type: From 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64'
    """
    def __init__(self, bits_per_symbol, preamble=None, log_ber_interval=10):
        ModulatorClassic.__init__(self, bits_per_symbol)
        gr.basic_block.__init__(self,
                                name="modulator_classic",
                                in_sig=None,
                                out_sig=None)
        if preamble is not numpy.ndarray:
            preamble = numpy.array(preamble)
        self.preamble = preamble
        self.log_ber_interval = log_ber_interval

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
        data_si = util_data.bits_to_integers(vec, self.bits_per_symbol)
        symbs = self.modulate(data_si).astype(numpy.complex64)
        self.message_port_pub(self.port_id_out,
                              pmt.cons(pmt.PMT_NIL,
                                       pmt.to_pmt(symbs)))
        t1 = time.time()
        self.logger.debug("classic mod {} handled {} bits in {} seconds".format(
                          self.uuid_str, vec.size, t1 - t0))

    def handle_update(self, pdu):
        self.ber_cnt += 1
        if self.preamble is not None and self.ber_cnt % self.log_ber_interval == 0:
            tag_dict = pmt.car(pdu)
            vec = pmt.to_python(pmt.cdr(pdu))
            ber = sum(numpy.abs(self.preamble - vec)) * 1.0 / self.preamble.size
            with open("ber_echo_{}.csv".format(self.uuid_str), "a") as f:
                f.write("{},{}\n".format(self.ber_cnt, ber))
