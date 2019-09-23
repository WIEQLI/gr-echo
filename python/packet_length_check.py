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
from gnuradio import gr
import pmt

class packet_length_check(gr.basic_block):
    """
    docstring for block packet_length_check
    """
    def __init__(self, length):
        """
        Packet Length Checker
        Sends every received PDU through the passthrough port, and only packets with the right length through the
        validated port
        :param length: Expected number of samples in the PDU
        """
        gr.basic_block.__init__(self,
            name="packet_length_check",
            in_sig=None,
            out_sig=None)

        self.length = length

        self.port_id_in = pmt.intern("in")
        self.port_id_passthrough = pmt.intern("passthrough")
        self.port_id_validated = pmt.intern("validated")
        self.port_id_failed = pmt.intern("failed")
        self.message_port_register_in(self.port_id_in)
        self.message_port_register_out(self.port_id_passthrough)
        self.message_port_register_out(self.port_id_validated)
        self.message_port_register_out(self.port_id_failed)

        self.set_msg_handler(self.port_id_in, self.handle_packet)

        self.npackets = 0
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")

    def handle_packet(self, pdu):
        self.npackets += 1
        tags = pmt.car(pdu)
        data = pmt.to_python(pmt.cdr(pdu))
        if data.size == self.length:
            self.message_port_pub(self.port_id_validated, pdu)
        else:
            self.message_port_pub(self.port_id_failed, pdu)
            self.logger.debug(
                "Packet {} failed validation with size {} [{}]".format(self.npackets,
                                                                       data.size, self.length))
        self.message_port_pub(self.port_id_passthrough, pdu)
