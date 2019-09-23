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

import numpy
import pmt
from gnuradio import gr


class preamble_insert(gr.basic_block):
    """
    docstring for block preamble_insert
    """
    def __init__(self, preamble=None, add_head=True, rm_head=False, rm_tail=False):
        """
        Inputs:
        :param preamble: list or 1d array of preamble bits
        :param add_head: insert preamble if true
        :param rm_head: remove preamble if true
        :param rm_tail: remove trailing len(preamble) bits if true
        """
        gr.basic_block.__init__(self,
                                name="preamble_insert",
                                in_sig=None,
                                out_sig=None)
        if preamble is None:
            raise Exception("Preamble must be provided")
        if preamble is not numpy.ndarray:
            preamble = numpy.array(preamble).astype(numpy.int32)
        assert len(preamble.shape) == 1, "Preamble must be a vector, not a matrix with a dimension of size 1"
        assert add_head or rm_head or rm_tail, "At least one operation must be True"
        self.preamble = preamble
        self.add_head = add_head
        self.rm_head = rm_head
        self.rm_tail = rm_tail

        self.port_in_id = pmt.intern("in")
        self.port_out_id = pmt.intern("out")
        self.message_port_register_in(self.port_in_id)
        self.message_port_register_out(self.port_out_id)
        self.set_msg_handler(self.port_in_id, self.handle_pdu)

        self.npackets = 0

    def handle_pdu(self, pdu):
        """Insert or remove the preamble from a pdu."""
        self.npackets += 1
        tags = pmt.car(pdu)
        data = pmt.to_python(pmt.cdr(pdu)).astype(numpy.int32)
        if self.add_head:
            data = numpy.concatenate([self.preamble, data])
        elif self.rm_head:
            data = data[self.preamble.size:]
        if self.rm_tail:
            data = data[:-self.preamble.size]
        # print("packet {}: {} in, {} out".format(self.npackets, data.size, data_out.size))
        if data.size > 0:
            self.message_port_pub(self.port_out_id,
                                  pmt.cons(tags, pmt.to_pmt(data)))

