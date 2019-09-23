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
import numpy
from gnuradio import gr, gr_unittest
from gnuradio import blocks
from gnuradio import channels
from echo_packet_wrapper import echo_packet_wrapper
from modulator_classic import modulator_classic
from demodulator_classic import demodulator_classic
from triggered_vector_source_b import triggered_vector_source_b as tvsb
from torch_echo.utils.util_data import get_random_bits, bits_to_integers
from echo_packet_unwrapper import echo_packet_unwrapper

class qa_echo_packet_unwrapper (gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()
        self.nfreq_samps = 256
        self.freqs = [11./256, 41./256, 93./256]
        self.corr_reps = 4
        self.channel = channels.channel_model(noise_voltage=0.01, frequency_offset=1e-5, epsilon=1.0,
                                              taps=[0.8, 0.1, 0.05, 0.01, 0.01, 0.005])
        self.channel_in = blocks.pdu_to_tagged_stream(blocks.complex_t, "frame")
        self.channel_out = blocks.tagged_stream_to_pdu(blocks.complex_t, "frame")
        self.tb.connect(self.channel_in, self.channel)
        self.tb.connect(self.channel, self.channel_out)

    def tearDown(self):
        self.channel = None
        self.channel_in = None
        self.channel_out = None
        self.tb = None

    def test_identity_channel_t(self):
        print("Identity Channel")
        body = get_random_bits(256)
        src = tvsb(body.astype(numpy.int8), None, True, "body")
        mod = modulator_classic("BPSK")
        wrapper = echo_packet_wrapper(self.nfreq_samps, self.freqs, self.corr_reps)
        unwrapper = echo_packet_unwrapper(self.nfreq_samps, self.freqs, self.corr_reps)
        demod = demodulator_classic("BPSK")
        sink = blocks.vector_sink_b()
        pdu2strm = blocks.pdu_to_tagged_stream(blocks.byte_t, "body")
        strm2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, "body")
        # Connect blocks
        self.tb.connect(src, strm2pdu)
        self.tb.connect(pdu2strm, sink)
        self.tb.msg_connect(strm2pdu, "pdus", mod, "bits")
        self.tb.msg_connect(mod, "symbols", wrapper, "body")
        self.tb.msg_connect(wrapper, "frame", unwrapper, "frame")
        self.tb.msg_connect(unwrapper, "body", demod, "symbols")
        self.tb.msg_connect(demod, "bits", pdu2strm, "pdus")
        # Run
        self.tb.start()
        while unwrapper.npackets < 1:
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        # check data
        result = sink.data()
        self.assertEqual(body.size, len(result), "Received {} bits, expected {}".format(len(result), body.size))
        for i in range(body.size):
            self.assertEqual(body[i], result[i],
                             "Received bit {} at {} does not match transmitted bit {}".format(body[i], i, result[i]))

    def test_awgn_channel_t(self):
        print("AWGN Channel")
        body = get_random_bits(256)
        src = tvsb(body.astype(numpy.int8), None, True, "body")
        mod = modulator_classic("BPSK")
        wrapper = echo_packet_wrapper(self.nfreq_samps, self.freqs, self.corr_reps)
        unwrapper = echo_packet_unwrapper(self.nfreq_samps, self.freqs, self.corr_reps)
        demod = demodulator_classic("BPSK")
        sink = blocks.vector_sink_b()
        pdu2strm = blocks.pdu_to_tagged_stream(blocks.byte_t, "body")
        strm2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, "body")
        # Connect blocks
        self.tb.connect(src, strm2pdu)
        self.tb.connect(pdu2strm, sink)
        self.tb.msg_connect(strm2pdu, "pdus", mod, "bits")
        self.tb.msg_connect(mod, "symbols", wrapper, "body")
        self.tb.msg_connect(wrapper, "frame", self.channel_in, "pdus")
        self.tb.msg_connect(self.channel_out, "pdus", unwrapper, "frame")
        self.tb.msg_connect(unwrapper, "body", demod, "symbols")
        self.tb.msg_connect(demod, "bits", pdu2strm, "pdus")
        # Run
        self.tb.start()
        while unwrapper.npackets < 1:
            print("wrapped:", wrapper.npackets)
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        # check data
        result = sink.data()
        self.assertEqual(len(result), body.size, "Received {} bits, expected {}".format(len(result), body.size))
        for i in range(body.size):
            self.assertEqual(body[i], result[i],
                             "Received bit {} at {} does not match transmitted bit {}".format(result[i], i, body[i]))


if __name__ == '__main__':
    gr_unittest.run(qa_echo_packet_unwrapper, "qa_echo_packet_unwrapper.xml")
