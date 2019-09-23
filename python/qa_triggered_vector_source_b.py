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
import numpy as np
from gnuradio import gr, gr_unittest
from gnuradio import blocks
from triggered_vector_source_b import triggered_vector_source_b as tvsb
from torch_echo.utils import util_data


class qa_triggered_vector_source(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()
        self.tag_key = "trigger"

    def tearDown(self):
        self.tb = None

    def test_unfiltered_t(self):
        print("Unfiltered Test")
        # Construct
        bits = util_data.get_random_bits(100).astype(np.int8)
        source = tvsb(data=bits, triggers=None, initial_trigger=True, tag_key=self.tag_key, debug_key="DEBUG")
        sink = blocks.vector_sink_b()
        strm2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, self.tag_key)
        # Connect
        self.tb.connect(source, sink)
        self.tb.connect(source, strm2pdu)
        self.tb.msg_connect(strm2pdu, "pdus", source, "trigger")
        # Confirm
        self.tb.start()
        cycles = 0
        while source.ntriggers < 3 and cycles < 50:
            time.sleep(0.1)
            cycles += 1
        self.tb.stop()
        self.tb.wait()
        ntrigs = source.ntriggers
        result = sink.data()
        self.assertGreater(ntrigs, 2, "Insufficient triggers {}".format(ntrigs))
        # The data from the final trigger might not have made it to the sink before we halted the top block,
        # so allow the last trigger's data to be missing
        self.assertEqual(len(result), ntrigs * len(bits) or len(result) == (ntrigs - 1) * len(bits),
                         "Incorrect samples out, actual {} != {} expected".format(len(result), ntrigs * len(bits)))
        for i, r in enumerate(result):
            self.assertEqual(r, bits[i % len(bits)], "Incorrect data out at index {}".format(i))

    def test_filtered_keep_t(self):
        print("Filtered Keep Test")
        bits = util_data.get_random_bits(100).astype(np.int8)
        source = tvsb(data=bits, triggers=["DEBUG"], initial_trigger=True, tag_key=self.tag_key, debug_key="DEBUG")
        sink = blocks.vector_sink_b()
        strm2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, self.tag_key)
        # Connect
        self.tb.connect(source, sink)
        self.tb.connect(source, strm2pdu)
        self.tb.msg_connect(strm2pdu, "pdus", source, "trigger")
        # Confirm
        self.tb.start()
        cycles = 0
        while source.ntriggers < 3 and cycles < 50:
            time.sleep(0.1)
            cycles += 1
        self.tb.stop()
        self.tb.wait()
        ntrigs = source.ntriggers
        result = sink.data()
        self.assertGreater(ntrigs, 2, "Insufficient number of triggers {}".format(ntrigs))
        for i, r in enumerate(result):
            self.assertEqual(r, bits[i % len(bits)], "Incorrect data out at index {}".format(i))

    def test_filtered_drop_t(self):
        print("Filtered Drop Test")
        bits = util_data.get_random_bits(100).astype(np.int8)
        source = tvsb(data=bits, triggers=["dropme"], initial_trigger=True, tag_key=self.tag_key, debug_key="DEBUG")
        sink = blocks.vector_sink_b()
        strm2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, self.tag_key)
        # Connect
        self.tb.connect(source, sink)
        self.tb.connect(source, strm2pdu)
        self.tb.msg_connect(strm2pdu, "pdus", source, "trigger")
        # Confirm
        self.tb.start()
        time.sleep(0.5)
        self.tb.stop()
        self.tb.wait()
        ntrigs = source.ntriggers
        self.assertEqual(ntrigs, 1, "Only one trigger should have been received")


if __name__ == '__main__':
    gr_unittest.run(qa_triggered_vector_source, "qa_triggered_vector_source.xml")
