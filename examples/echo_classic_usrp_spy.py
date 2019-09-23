#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Echo Classic Usrp Spy
# Generated: Thu May  2 18:46:08 2019
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio import zeromq
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import pmt
import echo
import time


class echo_classic_usrp_spy(gr.top_block):

    def __init__(self, beta_rrc=0.13, bits_per_symb=2, cfar_thresh=8, cfo_samps=512, corr_reps=2, log_interval=1, packet_len=512, samps_per_symb=2, spy_length=64, spy_thresh=0.1):
        gr.top_block.__init__(self, "Echo Classic Usrp Spy")

        ##################################################
        # Parameters
        ##################################################
        self.beta_rrc = beta_rrc
        self.bits_per_symb = bits_per_symb
        self.cfar_thresh = cfar_thresh
        self.cfo_samps = cfo_samps
        self.corr_reps = corr_reps
        self.log_interval = log_interval
        self.packet_len = packet_len
        self.samps_per_symb = samps_per_symb
        self.spy_length = spy_length
        self.spy_thresh = spy_thresh

        ##################################################
        # Variables
        ##################################################
        self.tx_gain = tx_gain = 4
        self.samp_rate = samp_rate = 500000
        self.rx_gain = rx_gain = 3
        self.cfo_freqs = cfo_freqs = 11./256, 43./256, 97./256
        self.body = body = [1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0]
        self.F_center = F_center = 1000000000

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_pull_msg_source_tx = zeromq.pull_msg_source("tcp://127.0.0.1:5555", 100, True)
        self.zeromq_pull_msg_source_rx = zeromq.pull_msg_source("tcp://127.0.0.1:5556", 100, True)
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(2),
        	),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(F_center, 0)
        self.uhd_usrp_source_0.set_gain(20, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_center_freq(F_center, 1)
        self.uhd_usrp_source_0.set_gain(rx_gain, 1)
        self.uhd_usrp_source_0.set_antenna("RX2", 1)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
        	",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        	"packet_len",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(F_center, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.echo_watchdog_0 = echo.watchdog(5)
        self.echo_preamble_insert_0_1 = echo.preamble_insert((body), 1, 0, 0)
        self.echo_preamble_insert_0_0 = echo.preamble_insert((body), 0, 1, 0)
        self.echo_preamble_insert_0 = echo.preamble_insert((body), 1, 0, 1)
        self.echo_packet_length_check_0 = echo.packet_length_check((2 * packet_len + spy_length) / bits_per_symb)
        self.echo_modulator_classic_spy_0 = echo.modulator_classic_spy(bits_per_symb, (body), log_interval, spy_length)
        self.echo_demodulator_classic_spy_0 = echo.demodulator_classic_spy(bits_per_symbol=bits_per_symb, block_length=1024,
              preamble=(body), log_ber_interval=log_interval,
              spy_length=spy_length, spy_threshold=spy_thresh)
        self.neural_to_classic_handler = echo.packet_handler(cfo_samps, corr_reps, (2 * packet_len + spy_length) / bits_per_symb, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.blocks_random_pdu_0 = blocks.random_pdu(packet_len, packet_len, chr(0x01), 1)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex*1)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_random_pdu_0, 'pdus'), (self.echo_preamble_insert_0_1, 'in'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_packet_length_check_0, 'in'))    
        self.msg_connect((self.echo_demodulator_classic_spy_0, 'corrupt'), (self.blocks_random_pdu_0, 'generate'))    
        self.msg_connect((self.echo_demodulator_classic_spy_0, 'bits'), (self.echo_preamble_insert_0, 'in'))    
        self.msg_connect((self.echo_demodulator_classic_spy_0, 'bits'), (self.echo_preamble_insert_0_0, 'in'))    
        self.msg_connect((self.echo_modulator_classic_spy_0, 'symbols'), (self.neural_to_classic_handler, 'wrap_in'))    
        self.msg_connect((self.echo_packet_length_check_0, 'failed'), (self.blocks_random_pdu_0, 'generate'))    
        self.msg_connect((self.echo_packet_length_check_0, 'validated'), (self.echo_demodulator_classic_spy_0, 'symbols'))    
        self.msg_connect((self.echo_packet_length_check_0, 'passthrough'), (self.echo_watchdog_0, 'in'))    
        self.msg_connect((self.echo_preamble_insert_0, 'out'), (self.echo_modulator_classic_spy_0, 'bits'))    
        self.msg_connect((self.echo_preamble_insert_0_0, 'out'), (self.echo_modulator_classic_spy_0, 'update'))    
        self.msg_connect((self.echo_preamble_insert_0_1, 'out'), (self.echo_modulator_classic_spy_0, 'bits'))    
        self.msg_connect((self.echo_watchdog_0, 'out'), (self.blocks_random_pdu_0, 'generate'))    
        self.msg_connect((self.zeromq_pull_msg_source_rx, 'out'), (self.uhd_usrp_source_0, 'command'))    
        self.msg_connect((self.zeromq_pull_msg_source_tx, 'out'), (self.uhd_usrp_sink_0, 'command'))    
        self.connect((self.neural_to_classic_handler, 0), (self.uhd_usrp_sink_0, 0))    
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_null_sink_0, 0))    
        self.connect((self.uhd_usrp_source_0, 1), (self.neural_to_classic_handler, 0))    

    def get_beta_rrc(self):
        return self.beta_rrc

    def set_beta_rrc(self, beta_rrc):
        self.beta_rrc = beta_rrc

    def get_bits_per_symb(self):
        return self.bits_per_symb

    def set_bits_per_symb(self, bits_per_symb):
        self.bits_per_symb = bits_per_symb

    def get_cfar_thresh(self):
        return self.cfar_thresh

    def set_cfar_thresh(self, cfar_thresh):
        self.cfar_thresh = cfar_thresh

    def get_cfo_samps(self):
        return self.cfo_samps

    def set_cfo_samps(self, cfo_samps):
        self.cfo_samps = cfo_samps

    def get_corr_reps(self):
        return self.corr_reps

    def set_corr_reps(self, corr_reps):
        self.corr_reps = corr_reps

    def get_log_interval(self):
        return self.log_interval

    def set_log_interval(self, log_interval):
        self.log_interval = log_interval

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len

    def get_samps_per_symb(self):
        return self.samps_per_symb

    def set_samps_per_symb(self, samps_per_symb):
        self.samps_per_symb = samps_per_symb

    def get_spy_length(self):
        return self.spy_length

    def set_spy_length(self, spy_length):
        self.spy_length = spy_length

    def get_spy_thresh(self):
        return self.spy_thresh

    def set_spy_thresh(self, spy_thresh):
        self.spy_thresh = spy_thresh

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)
        	

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 1)
        	

    def get_cfo_freqs(self):
        return self.cfo_freqs

    def set_cfo_freqs(self, cfo_freqs):
        self.cfo_freqs = cfo_freqs

    def get_body(self):
        return self.body

    def set_body(self, body):
        self.body = body

    def get_F_center(self):
        return self.F_center

    def set_F_center(self, F_center):
        self.F_center = F_center
        self.uhd_usrp_sink_0.set_center_freq(self.F_center, 0)
        self.uhd_usrp_source_0.set_center_freq(self.F_center, 0)
        self.uhd_usrp_source_0.set_center_freq(self.F_center, 1)


def argument_parser():
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")
    parser.add_option(
        "", "--beta-rrc", dest="beta_rrc", type="eng_float", default=eng_notation.num_to_str(0.13),
        help="Set RRC Beta [default=%default]")
    parser.add_option(
        "", "--bits-per-symb", dest="bits_per_symb", type="intx", default=2,
        help="Set Bits Per Symbol [default=%default]")
    parser.add_option(
        "", "--cfar-thresh", dest="cfar_thresh", type="eng_float", default=eng_notation.num_to_str(8),
        help="Set CFAR Threshold [default=%default]")
    parser.add_option(
        "", "--cfo-samps", dest="cfo_samps", type="intx", default=512,
        help="Set CFO Samples [default=%default]")
    parser.add_option(
        "", "--corr-reps", dest="corr_reps", type="intx", default=2,
        help="Set Correlator Repetitions [default=%default]")
    parser.add_option(
        "", "--log-interval", dest="log_interval", type="intx", default=1,
        help="Set Constellation Log Interval [default=%default]")
    parser.add_option(
        "", "--packet-len", dest="packet_len", type="intx", default=512,
        help="Set Packet Length [default=%default]")
    parser.add_option(
        "", "--samps-per-symb", dest="samps_per_symb", type="intx", default=2,
        help="Set Samples Per Symbol [default=%default]")
    parser.add_option(
        "", "--spy-length", dest="spy_length", type="intx", default=64,
        help="Set Spy Length [default=%default]")
    parser.add_option(
        "", "--spy-thresh", dest="spy_thresh", type="eng_float", default=eng_notation.num_to_str(0.1),
        help="Set Spy Threshold [default=%default]")
    return parser


def main(top_block_cls=echo_classic_usrp_spy, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    tb = top_block_cls(beta_rrc=options.beta_rrc, bits_per_symb=options.bits_per_symb, cfar_thresh=options.cfar_thresh, cfo_samps=options.cfo_samps, corr_reps=options.corr_reps, log_interval=options.log_interval, packet_len=options.packet_len, samps_per_symb=options.samps_per_symb, spy_length=options.spy_length, spy_thresh=options.spy_thresh)
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
