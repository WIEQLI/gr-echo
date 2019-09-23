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

import uuid
import threading
import numpy
import pmt
from gnuradio import gr


class watchdog(gr.basic_block):
    """
    docstring for block watchdog
    """
    def __init__(self, timeout):
        gr.basic_block.__init__(self,
            name="watchdog",
            in_sig=[],
            out_sig=[])

        self.timeout = timeout
        self.port_out = pmt.intern("out")
        self.port_in = pmt.intern("in")
        self.message_port_register_out(self.port_out)
        self.message_port_register_in(self.port_in)
        self.set_msg_handler(self.port_in, self.handle_msg)
        
        self.timer = threading.Timer(self.timeout, self.alarm)

        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.npackets = 0
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")

        self.timer.start()

    def handle_msg(self, pdu):
        self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.alarm)
        self.timer.start()

    def alarm(self):
        self.logger.debug("watchdog {} timed out")
        self.message_port_pub(self.port_out, pmt.cons(pmt.PMT_NIL, pmt.PMT_NIL))
        self.timer = threading.Timer(self.timeout, self.alarm)
        self.timer.start()

