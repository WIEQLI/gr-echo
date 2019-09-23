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
from EchoPacketWrapper import EchoPacketWrapper


class echo_packet_wrapper(gr.basic_block):
    """
    docstring for block echo_packet_wrapper
    """

    def __init__(self, samps_per_symb, beta_rrc, cfo_samps, cfo_freqs, corr_reps, 
                 body_samps=608):
        """
        Inputs:
        :param samps_per_symb: number of samples per symbol sent over the air
        :param beta_rrc: bandwidth expansion parameter for RRC filter
        :param cfo_samps: integer number of samples for the CFO correction header portion
        :param cfo_freqs: list of frequencies present in the CFO correction header, Hz
                          positive values only: freqs are added as cosines to mirror in negative frequency portion
        :param corr_reps: integer number of repetitions for correlation header Golay sequences
        """
        gr.basic_block.__init__(self,
                                name="echo_packet_wrapper",
                                in_sig=None,
                                out_sig=None)

        self.wrapper = EchoPacketWrapper(samps_per_symb=samps_per_symb, beta_rrc=beta_rrc,
                                         cfo_samps=cfo_samps, cfo_freqs=cfo_freqs,
                                         corr_repetitions=corr_reps)
        self.nsamples = self.wrapper.full_packet_length(body_samps) * samps_per_symb

        self.port_in_id = pmt.intern("body")
        self.port_out_id = pmt.intern("frame")
        self.message_port_register_in(self.port_in_id)
        self.message_port_register_out(self.port_out_id)
        self.set_msg_handler(self.port_in_id, self.handle_body)

        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.npackets = 0
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")

        # Send an all 0s pdu to prime the UHD block
        #zeros = numpy.zeros((self.nsamples,), dtype=numpy.complex64)
        #self.message_port_pub(self.port_out_id,
        #                      pmt.cons(pmt.PMT_NIL, pmt.to_pmt(zeros)))
        #self.message_port_pub(self.port_out_id,
        #                      pmt.cons(pmt.PMT_NIL, pmt.to_pmt(zeros)))

    def handle_body(self, pdu):
        t0 = time.time()
        self.npackets += 1
        tags = pmt.car(pdu)
        data = pmt.to_python(pmt.cdr(pdu))
        assert type(data) is numpy.ndarray
        assert type(data[0]) is numpy.complex64
        frame = self.wrapper.wrap(data)
        self.message_port_pub(self.port_out_id,
                              pmt.cons(tags, pmt.to_pmt(frame)))
        t1 = time.time()
        self.logger.debug("packet wrap {} handled {} symbols in {} seconds".format(
                          self.uuid_str, data.size, t1 - t0))
