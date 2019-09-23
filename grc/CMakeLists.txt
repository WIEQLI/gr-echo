# Copyright 2011 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

install(FILES
    echo_modulator_classic.xml
    echo_demodulator_classic.xml
    echo_modulator_neural.xml
    echo_demodulator_neural.xml
    echo_triggered_vector_source_b.xml
    echo_echo_packet_wrapper.xml
    echo_echo_packet_unwrapper.xml
    echo_echo_packet_detect.xml
    echo_rand_zeros_extend_c.xml
    echo_packet_length_check.xml
    echo_pdu_complex_to_ettus_tagged_stream.xml
    echo_preamble_insert.xml
    echo_packet_handler.xml
    echo_add_message_to_stream_async.xml
    echo_watchdog.xml
    echo_demodulator_classic_spy.xml
    echo_demodulator_neural_spy.xml
    echo_modulator_classic_spy.xml
    echo_modulator_neural_spy.xml
    echo_echo_mod_demod.xml DESTINATION share/gnuradio/grc/blocks
)
