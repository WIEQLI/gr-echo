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


class triggered_vector_source_b(gr.sync_block):
    """
    docstring for block triggered_vector_source_b
    """

    def __init__(self, data, triggers, initial_trigger=False, tag_key="trigger", debug_key=None):
        """
        Inputs:
        :param data: (N,) np.int8 array containing data to send with each trigger
        :param triggers: list of strings to act as filters for triggering
        :param initial_trigger: flag telling block to act as if it received a trigger on startup
        :param tag_key: optional gr tag to send with each set of data, value is len(data)
        """
        gr.sync_block.__init__(self,
                               name="triggered_vector_source_b",
                               in_sig=None,
                               out_sig=[numpy.int8])
        # Data
        if type(data) is not numpy.ndarray:
            data = numpy.array(data)
        self.data = data.reshape(-1, 1)
        self.queue = numpy.array([[]], dtype=numpy.int8).reshape(-1, 1)
        self.initial_trigger = initial_trigger
        # Tag
        if tag_key is not None:
            self.tag = gr.tag_t()
            self.tag.key = pmt.intern(tag_key)
            self.tag.value = pmt.from_long(len(self.data))
            self.tag.offset = 0
            self.tag.srcid = pmt.intern("triggered_vector_source_b")
            if debug_key is not None:
                self.debug_tag = gr.tag_t()
                self.debug_tag.key = pmt.intern(debug_key)
                self.debug_tag.value = pmt.PMT_NIL
                self.debug_tag.offset = 0
                self.debug_tag.srcid = pmt.intern("DEBUG triggered_vector_source_b")
            else:
                self.debug_tag = None
        else:
            self.tag = None
            self.debug_tag = None
        # Triggers
        if triggers is not None and len(triggers) >= 1:
            self.trigger_tags = triggers
        else:
            self.trigger_tags = None
        self.trig_port = pmt.intern("trigger")
        self.message_port_register_in(self.trig_port)
        self.set_msg_handler(self.trig_port, self.trigger)
        self.ntriggers = initial_trigger

    def work(self, input_items, output_items):
        if self.initial_trigger:
            self.add_tags()
            self.queue = numpy.concatenate([self.queue, self.data])
            self.initial_trigger = False
        out = output_items[0]
        nitems = min(len(out), len(self.queue))
        out[0:nitems] = self.queue[0:nitems].squeeze()
        self.queue = self.queue[nitems:]
        return nitems

    def trigger(self, pdu):
        if self.trigger_tags is None:
            self.add_tags()
            self.queue = numpy.concatenate([self.queue, self.data])
            self.ntriggers += 1
        else:
            tags = pmt.to_python(pmt.car(pdu))
            if tags is not None:
                for t in tags:
                    if t in self.trigger_tags:
                        self.add_tags()
                        self.queue = numpy.concatenate([self.queue, self.data])
                        self.ntriggers += 1

    def tag_offset(self):
        return self.nitems_written(0) + len(self.queue)
    
    def add_tags(self):
        if self.tag is not None:
            self.tag.offset = self.tag_offset()
            self.add_item_tag(0, self.tag)
        if self.debug_tag is not None:
            self.debug_tag.offset = self.tag_offset() + 1
            self.add_item_tag(0, self.debug_tag)
