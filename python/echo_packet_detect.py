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
from DSPUtil import rrc_decimate_fft


class echo_packet_detect(gr.sync_block):
    """Docstring for block echo_packet_detect."""

    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    COMPLETE = "COMPLETE"

    def __init__(self, samps_per_symb, beta_rrc, cfo_samps, corr_reps, body_size, threshold=10):
        """
        Packet detector.

        :param samps_per_symb: number of samples per symbol sent over the air
        :param beta_rrc: bandwidth expansion parameter for RRC filter
        :param cfo_samps: number of sample in CFO estimation field
        :param corr_reps: number of repetitions of Golay sequence in channel estimation fields
        :param body_size: number of samples in the body of a packet
        :param threshold: threshold for detection of packet by Golay correlation
                               relative to normalized perfect correlation of 1.0
        """
        gr.sync_block.__init__(self,
                               name="echo_packet_detect",
                               in_sig=[np.complex64],
                               out_sig=None)

        self.samps_per_symb = samps_per_symb
        self.beta_rrc = beta_rrc
        self.cfo_samps = cfo_samps
        self.corr_reps = corr_reps
        self.body_size = body_size
        self.threshold = threshold
        self.wrapper = EchoPacketWrapper(samps_per_symb=samps_per_symb, beta_rrc=beta_rrc,
                                         cfo_samps=cfo_samps, cfo_freqs=[0.25],
                                         corr_repetitions=corr_reps)

        # Require that a full cfo field and channel est field be present at the same time
        self.nhistory = (self.wrapper.corr_samps) * self.samps_per_symb - 1
        self.set_history(self.nhistory + 1)  # GNURadio actually gives you N-1 history items
        self.drop_history = 0
        self.state = self.SEARCHING
        self.pkt_buf = None  # Buffer to build packets in when broken across calls to work
        self.missing_samples = 0  # Nonzero when the tail of a packet is missing from a work call

        self.port_id_out = pmt.intern("frame")
        self.message_port_register_out(self.port_id_out)

        self.npackets = 0
        self.uuid = uuid.uuid4()
        self.uuid_str = str(self.uuid)[-6:]

        self.logger = gr.logger("log_debug")
        self.logger.set_level("DEBUG")
        self.logger.info("DETECTOR {}: Packet Size = {} samples".format(
                         self.uuid_str, self.wrapper.full_packet_length(self.body_size)))

    def __repr__(self):
        """Print summary of state."""
        s = "echo_packet_detect {}:\n".format(self.uid)
        s += "\tnpackets {}\n".format(self.npackets)
        s += "\tstate {}\n".format(self.state)
        if self.pkt_buf is None:
            s += "\tpkt_buf is None"
        else:
            s += "\tpkt_buf size is {}\n".format(self.pkt_buf.size)
            s += "\tmissing samples is {}".format(self.missing_samples)
        return s

    def work(self, input_items, output_items):
        """Handle calls from the gnuradio scheduler."""
        t0 = time.time()
        in0 = input_items[0]
        includes_history = True
        # print("WORK {} {}: entered work with {} new samples".format(
        #       self.uid, self.npackets, input_items[0].size - self.nhistory))
        # # If we're in the middle of a packet we can drop the history samples
        if self.state == self.FOUND:
            in0 = in0[self.nhistory:]
            # print("WARN {} {}: state is FOUND, removing history".format(self.uid, self.npackets))
            includes_history = False
        if self.state == self.SEARCHING and self.drop_history > 0:
            in0 = in0[self.drop_history:]
            # print("WARN {} {}: dropping {} historical samples".format(self.uid, self.npackets,
            #                                                           self.drop_history))
            # Dropped history decreases by the number of new samples until reaching 0
            new_drop = max(0, self.drop_history - (input_items[0].size - self.nhistory))
            self.drop_history = new_drop
            # print("\tnew drop count {}".format(new_drop))
        # Remove history that has already been used in previous work calls
        # in0 = in0[self.nhistory - self.keep_nhistory:]
        while in0.size > 0:
            # print("WORK {} {}: entered while loop".format(self.uid, self.npackets))
            # Begin work loop
            if self.state == self.SEARCHING:
                # print("\t{} SEARCHING".format(self.uid))
                in0 = self.search(in0, includes_history)
                # print("\t{}".format(in0.size))
                includes_history = False
            if self.state == self.FOUND:
                # print("\t{} FOUND".format(self.uid))
                in0 = self.extract(in0)
                # print("\t{}".format(in0.size))
                includes_history = False
            if self.state == self.COMPLETE:
                pkt = rrc_decimate_fft(self.pkt_buf, self.beta_rrc, self.samps_per_symb)
                self.message_port_pub(self.port_id_out,
                                      pmt.cons(pmt.to_pmt({}), pmt.to_pmt(pkt)))
                # self.logger.info("DETECTOR {}: Packet {} has size {}".format(
                #                  self.uid, self.npackets, self.pkt_buf.size))
                # print("DETECTOR {}: Packet {} has size {}".format(
                #       self.uid, self.npackets, pkt.size))
                # np.save("packet_{}_{}".format(self.npackets, self.uid), self.pkt_buf)
                self.pkt_buf = None
                self.npackets += 1
                self.state = self.SEARCHING
                # We don't want to accidentally grab the tail end of a previous packet that's kept
                # in the history buffer
                self.drop_history = max(0, self.nhistory - in0.size)
                includes_history = False
        t1 = time.time()
        # self.logger.debug(
        #     "packet detect {} completed work() on {} samples in {} seconds".format(
        #         self.uuid_str, input_items[0].size, t1 - t0))
        return input_items[0].size - self.nhistory

    def search(self, samps, includes_history):
        """Search for the start of a packet."""
        # We might be able to skip the first cfo_samps samples, but better safe than sorry...
        # Also by using FFTs for convolution the additional overhead is minimal
        delay = self.wrapper.find_channel_estimate_field(samps, cfar_threshold=self.threshold,
                                                         do_plot=False)
        if delay is None:
            return samps[0:0]
        else:
            self.state = self.FOUND
            # np.save("detection_{}_{}".format(self.npackets, self.uuid_str), samps)
            return samps[delay:]

    def extract(self, samps):
        """Extract packet data once one has been found."""
        if self.pkt_buf is None:
            self.missing_samples = (self.wrapper.full_packet_length(body_samps=self.body_size) *
                                    self.samps_per_symb)
            # TODO: pre-allocate full body here and fill it using a pointer
            self.pkt_buf = np.zeros((0,), dtype=np.complex64)
        # If we can grab the full packet with the data we have, do so
        if self.missing_samples <= samps.size:
            self.pkt_buf = np.concatenate([self.pkt_buf, samps[0:self.missing_samples]])
            n_taken = self.missing_samples
            self.missing_samples = 0
            self.state = self.COMPLETE
            return samps[n_taken:]
        # Otherwise get what we can until the next call to work
        else:
            self.pkt_buf = np.concatenate([self.pkt_buf, samps])
            self.missing_samples -= samps.size
            # print("WARN {} {}: missing {} samples".format(
            #       self.uid, self.npackets, self.missing_samples))
            return samps[0:0]
