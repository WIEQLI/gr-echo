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
from matplotlib import pyplot as plt
from gnuradio import gr, gr_unittest
from gnuradio import blocks
from echo_packet_wrapper import echo_packet_wrapper
from modulator_classic import modulator_classic
from triggered_vector_source_b import triggered_vector_source_b as tvsb
from torch_echo.utils.util_data import get_random_bits, bits_to_integers
from EchoPacketWrapper import EchoPacketWrapper

class qa_echo_packet_wrapper (gr_unittest.TestCase):

    def setUp (self):
        self.tb = gr.top_block ()
        self.nfreq_samps = 256
        self.freqs = [11./256, 41./256, 93./256]
        print("Norm freqs: {}".format(self.freqs))
        self.corr_reps = 4

    def tearDown (self):
        self.tb = None

    def plot_outcome(self, data):
        cfo = data[0:self.nfreq_samps]
        corr = data[self.nfreq_samps:self.nfreq_samps + self.corr_reps*256] / numpy.exp(0.5j*numpy.pi*numpy.arange(256))
        Ga, Gb = EchoPacketWrapper.golay_80211ad(128)
        # Plot CFO
        plt.plot(numpy.arange(-0.5, 0.5, 1./256), numpy.abs(numpy.fft.fftshift(numpy.fft.fft(cfo))))
        plt.title("CFO")
        plt.xlabel("Norm Freq")
        plt.show()
        # Plot Correlation
        mod = numpy.exp(-.5j * numpy.pi * numpy.arange(128))
        plt.plot(numpy.correlate(corr[:128], Ga * mod, 'same'), label='Ga')
        plt.plot(numpy.correlate(corr[128:256], Gb * mod, 'same'), label='Gb')
        corr_sum = numpy.correlate(corr[:128], Ga * mod, 'same') + numpy.correlate(corr[128:256], Gb * mod, 'same')
        plt.plot(range(128), corr_sum, label='Sum')
        plt.xlabel('Correlation lag')
        plt.title("Golay Correlator")
        plt.legend()
        plt.show()

    def test_001_t(self):
        print("Enable plotting to manually verify result of wrapping")
        body = get_random_bits(256)
        src = tvsb(body.astype(numpy.int8), None, True, "body")
        mod = modulator_classic("BPSK")
        wrapper = echo_packet_wrapper(self.nfreq_samps, self.freqs, self.corr_reps)
        sink = blocks.vector_sink_c()
        pdu2strm = blocks.pdu_to_tagged_stream(blocks.complex_t, "frame")
        strm2pdu = blocks.tagged_stream_to_pdu(blocks.byte_t, "body")
        # Connect blocks
        self.tb.connect(src, strm2pdu)
        self.tb.connect(pdu2strm, sink)
        self.tb.msg_connect(strm2pdu, "pdus", mod, "bits")
        self.tb.msg_connect(mod, "symbols", wrapper, "body")
        self.tb.msg_connect(wrapper, "frame", pdu2strm, "pdus")
        # Run
        self.tb.start()
        while wrapper.npackets < 1:
            time.sleep(0.1)
        self.tb.stop()
        self.tb.wait()
        # check data
        result = sink.data()
        # self.plot_outcome(result)
        self.assertEqual(len(result), self.nfreq_samps + 2*self.corr_reps*256 + 256, "Incorrect number of output bits")


if __name__ == '__main__':
    gr_unittest.run(qa_echo_packet_wrapper, "qa_echo_packet_wrapper.xml")
