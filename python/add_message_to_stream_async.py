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

import time
import uuid
import numpy
import pmt
from gnuradio import gr

class add_message_to_stream_async(gr.sync_block):
    """
    docstring for block add_message_to_stream_async
    """
    def __init__(self):
        gr.sync_block.__init__(self,
            name="add_message_to_stream_async",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.message_port_register_in(pmt.intern("msgs"))
        self.set_msg_handler(pmt.intern("msgs"), self.handle_message)

        self.msg_queue = numpy.zeros((0,), dtype=numpy.complex64)
        self.msgs_received = 0

        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]
        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("async_msg2strm {}".format(self.uuid_str))

    def work(self, input_items, output_items):
        t0 = time.time()
        in0 = input_items[0]
        out = output_items[0]
        out[:] = in0
        to_take = min(out.size, self.msg_queue.size)
        if to_take > 0:
            out[:to_take] += self.msg_queue[:to_take]
            self.msg_queue = self.msg_queue[to_take:]
        t1 = time.time()
        # self.logger.debug("async_msg2strm {} passed {} samples in {} seconds".format(
        #                   self.uuid_str, in0.size, t1 - t0))
        # if to_take > 0:
        #     self.logger.debug("async_msg2strm {} added {} samples to the stream".format(
        #                       self.uuid_str, to_take))
        return len(output_items[0])

    def handle_message(self, pdu):
        data = pmt.to_python(pmt.cdr(pdu))
        assert data.dtype == numpy.complex64, "Can only handle complex64 data"
        self.msg_queue = numpy.concatenate((self.msg_queue, numpy.ravel(data)))
        # self.logger.debug("async_msg2strm {} received a {} sample pdu".format(
        #                   self.uuid_str, data.size))
