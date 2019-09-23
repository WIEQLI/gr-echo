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

import time
from gnuradio import gr, gr_unittest
from gnuradio import blocks
import pmt
from demodulator_neural import demodulator_neural
from modulator_classic import modulator_classic
from torch_echo.utils.util_data import get_random_bits

class qa_demodulator_neural (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()
        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32e3
        self.preamble = preamble = get_random_bits(64*3)
        self.npreamble = len(self.preamble)
        # self.modulation = modulation = "QPSK"
        self.nbits = 16*3
        self.bits = get_random_bits(self.nbits)
        self.tag_preamble = gr.tag_t()
        self.tag_preamble.key = pmt.intern("pkt")
        self.tag_preamble.value = pmt.from_long(self.npreamble)
        self.tag_preamble.offset = 0
        self.tag_body = gr.tag_t()
        self.tag_body.key = pmt.intern("pkt")
        self.tag_body.value = pmt.from_long(self.nbits)

        ##################################################
        # Blocks
        ##################################################
        # self.modulator = modulator_classic(modulation)
        # self.demodulator = demodulator_neural(seed=0, hidden_layers=(64, ), bits_per_symbol=2, preamble=preamble)
        self.vector_source_preamble = blocks.vector_source_b(preamble, True, 1, [self.tag_preamble])
        self.vector_source_body = blocks.vector_source_b(self.bits, True, 1, [self.tag_body])
        self.tags_mux = blocks.tagged_stream_mux(gr.sizeof_char*1, "pkt")
        self.strm_to_pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, "pkt")
        self.pdu_to_strm = blocks.pdu_to_tagged_stream(blocks.byte_t, "pkt")
        self.vector_sink = blocks.vector_sink_b()

    def tearDown (self):
        self.tb = None
        self.vector_source_preamble = None
        self.vector_source_body = None
        self.vector_sink = None
        self.tags_mux = None
        self.strm_to_pdu = None
        self.pdu_to_strm = None

    def connect_tb(self, mod, demod):
        self.tb.connect(self.vector_source_preamble, (self.tags_mux, 0))
        self.tb.connect(self.vector_source_body, (self.tags_mux, 1))
        self.tb.connect(self.tags_mux, self.strm_to_pdu)
        self.tb.msg_connect(self.strm_to_pdu, "pdus", mod, "bits")
        self.tb.msg_connect(mod, "symbols", demod, "symbols")
        self.tb.msg_connect(demod, "bits", self.pdu_to_strm, "pdus")
        self.tb.connect(self.pdu_to_strm, self.vector_sink)

    def check_result(self, data):
        errs = 0
        if len(data) >= self.nbits+self.npreamble:
            for i in range(self.nbits + self.npreamble):
                j = i + len(data) - (self.nbits + self.npreamble)
                if i < self.npreamble:
                    errs += self.preamble[i] != data[j]
                else:
                    errs += self.bits[i - self.npreamble] != data[j]
        self.assertEqual(errs, 0, "Demodulation incorrect, {} errors found in last iteration".format(errs))
        return errs

    def test_bpsk_t (self):
        print("Testing BPSK learning...")
        # run tests
        mod = modulator_classic("BPSK")
        demod = demodulator_neural(seed=0, hidden_layers=(64,), bits_per_symbol=mod.bits_per_symbol,
                                   preamble=self.preamble)
        self.connect_tb(mod, demod)
        self.tb.start()
        while demod.packet_cnt < 75:
            # print("{} packets...".format(demod.packet_cnt))
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        data = self.vector_sink.data()
        self.check_result(data)
        # demod.visualize_decision_boundary(20)

    def test_qpsk_t (self):
        print("Testing QPSK learning...")
        # run tests
        mod = modulator_classic("QPSK")
        demod = demodulator_neural(0, (64,), mod.bits_per_symbol, self.preamble)
        self.connect_tb(mod, demod)
        self.tb.start()
        while demod.packet_cnt < 100:
            # print("{} packets...".format(demod.packet_cnt))
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        data = self.vector_sink.data()
        self.check_result(data)
        # demod.visualize_decision_boundary(20)

    def test_8psk_t (self):
        print("Testing 8PSK learning...")
        # run tests
        mod = modulator_classic("8PSK")
        demod = demodulator_neural(0, (64,), mod.bits_per_symbol, self.preamble)
        self.connect_tb(mod, demod)
        self.tb.start()
        while demod.packet_cnt < 250:
            # print("{} packets...".format(demod.packet_cnt))
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        data = self.vector_sink.data()
        self.check_result(data)
        # demod.visualize_decision_boundary(20)

    def test_qam16_t (self):
        print("Testing QAM16 learning...")
        # run tests
        mod = modulator_classic("QAM16")
        demod = demodulator_neural(0, (64,), mod.bits_per_symbol, self.preamble)
        self.connect_tb(mod, demod)
        self.tb.start()
        while demod.packet_cnt < 1000:
            # print("{} packets...".format(demod.packet_cnt))
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        data = self.vector_sink.data()
        self.check_result(data)
        # demod.visualize_decision_boundary(20)

    # def test_qam64_t (self):
    #     print("Testing QAM64 learning...")
    #     # run tests
    #     mod = modulator_classic("QAM64")
    #     demod = demodulator_neural(0, (64, 16), mod.bits_per_symbol, self.preamble)
    #     self.connect_tb(mod, demod)
    #     self.tb.start()
    #     while demod.packet_cnt < 100:
    #         # print("{} packets...".format(demod.packet_cnt))
    #         time.sleep(0.1)
    #     self.tb.stop()
    #     self.tb.wait()
    #     data = self.vector_sink.data()
    #     self.check_result(data)
    #     # demod.visualize_decision_boundary(20)



if __name__ == '__main__':
    gr_unittest.run(qa_demodulator_neural, "qa_demodulator_neural.xml")

