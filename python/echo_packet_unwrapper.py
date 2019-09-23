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

import numpy as np
import time
import uuid
import pmt
from gnuradio import gr

from EchoPacketWrapper import EchoPacketWrapper


class echo_packet_unwrapper(gr.basic_block):
    """
    docstring for block echo_packet_unwrapper
    """
    def __init__(self, samps_per_symb, beta_rrc, cfo_samps, cfo_freqs, corr_reps):
        """
        Inputs:
        :param samps_per_symb: number of samples per symbol sent over the air
        :param beta_rrc: bandwidth expansion parameter for RRC filter
        :param cfo_samps: integer number of samples for the CFO correction header portion
        :param cfo_freqs: list of frequencies present in the CFO correction header, Hz
                          positive values only: freqs are added as cosines to mirror in negative
                          frequency portion
        :param corr_reps: integer number of repetitions for correlation header Golay sequences
        """
        gr.basic_block.__init__(self,
                                name="echo_packet_unwrapper",
                                in_sig=None,
                                out_sig=None)

        self.wrapper = EchoPacketWrapper(samps_per_symb=samps_per_symb, beta_rrc=beta_rrc,
                                         cfo_samps=cfo_samps, cfo_freqs=cfo_freqs,
                                         corr_repetitions=corr_reps)

        self.chan0 = np.zeros((128,), dtype=np.complex64)
        self.chan0[64] = 1.0 + 0.0j

        self.port_in_id = pmt.intern("frame")
        self.port_out_id = pmt.intern("body")
        self.message_port_register_in(self.port_in_id)
        self.message_port_register_out(self.port_out_id)
        self.set_msg_handler(self.port_in_id, self.handle_frame)

        self.npackets = 0
        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")

    def handle_frame(self, pdu):
        t0 = time.time()
        self.npackets += 1
        tags = pmt.car(pdu)
        data = pmt.to_python(pmt.cdr(pdu))
        assert type(data) is np.ndarray
        assert type(data[0]) is np.complex64

        body, _ = self.wrapper.unwrap(data, do_plot=False)
        body = body.astype(np.complex64)

        self.message_port_pub(self.port_out_id,
                              pmt.cons(tags, pmt.to_pmt(body)))
        t1 = time.time()
        self.logger.debug("packet unwrap {} handled {} symbols in {} seconds".format(
                          self.uuid_str, data.size, t1 - t0))
