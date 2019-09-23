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
import pmt
from gnuradio import gr


class rand_zeros_extend_c(gr.basic_block):
    """
    docstring for block rand_zeros_extend_c
    """
    def __init__(self, prepend_min, prepend_max, append_min, append_max):
        gr.basic_block.__init__(self,
            name="rand_zeros_extend_c",
            in_sig=None,
            out_sig=None)
        self.prepend_min = prepend_min
        self.prepend_max = prepend_max
        self.append_min = append_min
        self.append_max = append_max

        self.port_id_in = pmt.intern("frame")
        self.port_id_out = pmt.intern("extended")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_out(self.port_id_out)
        self.set_msg_handler(self.port_id_in, self.handle_frame)

        self.npackets = 0

        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")

    def handle_frame(self, pdu):
        self.npackets += 1
        tags = pmt.car(pdu)
        data = pmt.to_python(pmt.cdr(pdu))
        npre = numpy.random.randint(self.prepend_min, self.prepend_max + 1)
        npost = numpy.random.randint(self.append_min, self.append_max + 1)
        data = numpy.concatenate([numpy.zeros(npre, dtype=numpy.complex64),
                                  data,
                                  numpy.zeros(npost, dtype=numpy.complex64)])
        self.message_port_pub(self.port_id_out,
                              pmt.cons(tags, pmt.to_pmt(data)))
        # self.logger.debug("Packet {} sent with ({}, {}) extension".format(self.npackets, npre, npost))
