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

from gnuradio import gr
from gnuradio import blocks
from gnuradio import digital
import pmt
from echo_packet_wrapper import echo_packet_wrapper as EPW
from echo_packet_unwrapper import echo_packet_unwrapper as EPU
from echo_packet_detect import echo_packet_detect as EPD
from add_message_to_stream_async import add_message_to_stream_async as MessageAddAsync


class packet_handler(gr.hier_block2):
    """
    docstring for block packet_handler
    """
    def __init__(self, cfo_samps, corr_reps, body_size, cfar_thresh, samps_per_symb, beta_rrc, cfo_freqs):
        gr.hier_block2.__init__(self,
                                "packet_handler",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex),  # Input signature
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))  # Output signature

        # Define blocks and connect them
        self.wrapper = EPW(cfo_samps=cfo_samps, cfo_freqs=cfo_freqs, corr_reps=corr_reps,
                           samps_per_symb=samps_per_symb, beta_rrc=beta_rrc)
        self.unwrapper = EPU(cfo_samps=cfo_samps, cfo_freqs=cfo_freqs, corr_reps=corr_reps,
                             samps_per_symb=samps_per_symb, beta_rrc=beta_rrc)
        self.detector = EPD(cfo_samps=cfo_samps, corr_reps=corr_reps,
                            samps_per_symb=samps_per_symb, beta_rrc=beta_rrc,
                            threshold=cfar_thresh, body_size=body_size)

        self.pdu2strm = blocks.pdu_to_tagged_stream(blocks.complex_t, "packet_len")
        self.set_min_output_buffer(8192)
        self.pdu2strm.set_min_output_buffer(8192)
        # TODO: Shape bursts with head and tail phasing samples and amplitude envelope
        # self.burst_shaper = digital.burst_shaper_cc((([])), 128, 128, True, "packet_len")

        # Create message ports
        self.port_wrap_in = pmt.intern("wrap_in")
        self.port_unwrap_out = pmt.intern("unwrap_out")
        self.primitive_message_port_register_hier_in(self.port_wrap_in)
        self.primitive_message_port_register_hier_out(self.port_unwrap_out)

        # Connect all ports
        # Inputs
        self.connect(self, self.detector)
        self.msg_connect(self, self.port_wrap_in, self.wrapper, pmt.intern("body"))
        # Outputs
        self.connect((self.pdu2strm, 0), (self, 0))
        self.msg_connect(self.unwrapper, pmt.intern("body"), self, self.port_unwrap_out)
        # Internal
        self.msg_connect(self.wrapper, pmt.intern("frame"), self.pdu2strm, pmt.intern("pdus"))
        self.msg_connect(self.detector, pmt.intern("frame"), self.unwrapper, pmt.intern("frame"))
