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

import pmt
import time
from gnuradio import gr, gr_unittest
from gnuradio import blocks
from demodulator_classic import demodulator_classic
from modulator_classic import modulator_classic
from torch_echo.utils.util_data import get_random_bits

class qa_demodulator_classic (gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_t(self):
        pkt_tag = gr.tag_t()
        pkt_tag.key = pmt.intern("pkt")
        mod_types = ('BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64')
        for mod_type in mod_types:
            print("Testing {}...".format(mod_type))
            modulator = modulator_classic(mod_type)
            demodulator = demodulator_classic(mod_type, 200)
            src_data = get_random_bits(1000 * modulator.bits_per_symbol)
            pkt_tag.value = pmt.to_pmt(len(src_data))
            src = blocks.vector_source_b(src_data, False, 1, [pkt_tag])
            snk = blocks.vector_sink_b()
            tag2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, "pkt")
            pdu2tag = blocks.pdu_to_tagged_stream(blocks.byte_t, "pkt")
            self.tb.connect(src, tag2pdu)
            self.tb.msg_connect(tag2pdu, "pdus", modulator, "bits")
            self.tb.msg_connect(modulator, "symbols", demodulator, "symbols")
            self.tb.msg_connect(demodulator, "bits", pdu2tag, "pdus")
            self.tb.connect(pdu2tag, snk)
            self.tb.start()
            # check data
            while demodulator.packet_cnt < 1:
                time.sleep(0.1)
            self.tb.stop()
            self.tb.wait()
            result_data = snk.data()
            for i in range(1000 * modulator.bits_per_symbol):
                self.assertEqual(src_data[i], result_data[i])


if __name__ == '__main__':
    gr_unittest.run(qa_demodulator_classic, "qa_demodulator_classic.xml")
