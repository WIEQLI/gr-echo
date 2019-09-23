#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Echo Single-USRP Flowgraph
# Author: Josh Sanz
# Generated: Fri Jul 19 18:16:58 2019
##################################################

from gnuradio import blocks
from gnuradio import channels
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


class echo_single_usrp(gr.top_block):

    def __init__(self, beta_rrc=0.13, bits_per_symb=2, cfar_thresh=50, cfo_samps=512, corr_reps=2, demod_hidden_layers=[50], demod_init_weights="", demod_seed=341133, demodtype="neural", lambda_center=1, log_interval=1, max_amplitude=0.707, mod_hidden_layers=[50], mod_init_weights="", mod_seed=341132, modtype="neural", packet_len=256, rx_gain=4, samps_per_symb=2, shared_preamble="", spy_thresh=0.1, tx_gain=3):
        gr.top_block.__init__(self, "Echo Single-USRP Flowgraph")

        ##################################################
        # Parameters
        ##################################################
        self.beta_rrc = beta_rrc
        self.bits_per_symb = bits_per_symb
        self.cfar_thresh = cfar_thresh
        self.cfo_samps = cfo_samps
        self.corr_reps = corr_reps
        self.demod_hidden_layers = demod_hidden_layers
        self.demod_init_weights = demod_init_weights
        self.demod_seed = demod_seed
        self.demodtype = demodtype
        self.lambda_center = lambda_center
        self.log_interval = log_interval
        self.max_amplitude = max_amplitude
        self.mod_hidden_layers = mod_hidden_layers
        self.mod_init_weights = mod_init_weights
        self.mod_seed = mod_seed
        self.modtype = modtype
        self.packet_len = packet_len
        self.rx_gain = rx_gain
        self.samps_per_symb = samps_per_symb
        self.shared_preamble = shared_preamble
        self.spy_thresh = spy_thresh
        self.tx_gain = tx_gain

        ##################################################
        # Variables
        ##################################################
        self.spy_length = spy_length = 64 * bits_per_symb
        self.samp_rate = samp_rate = 500000
        self.full_len = full_len = 2 * packet_len + (spy_length + 96) / bits_per_symb
        self.cfo_freqs = cfo_freqs = 11./256, 43./256, 97./256
        self.F_center = F_center = 1000000000

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_pull_msg_source_1 = zeromq.pull_msg_source("tcp://127.0.0.1:5556", 100, True)
        self.zeromq_pull_msg_source_0 = zeromq.pull_msg_source("tcp://127.0.0.1:5555", 100, True)
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(2),
        	),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(F_center, 0)
        self.uhd_usrp_source_0.set_gain(rx_gain, 0)
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
        self.echo_watchdog_0 = echo.watchdog(1)
        self.echo_echo_mod_demod_0 = echo.echo_mod_demod(packet_len * bits_per_symb, shared_preamble, bits_per_symb, 
              modtype, demodtype, 
              mod_seed=mod_seed, demod_seed=demod_seed, 
              mod_hidden_layers=(mod_hidden_layers), demod_hidden_layers=(demod_hidden_layers), 
              mod_init_weights=mod_init_weights, demod_init_weights=demod_init_weights,
              log_interval=log_interval, spy_length=spy_length, spy_threshold=0.1, 
              max_amplitude=max_amplitude, lambda_center=lambda_center,
              _alias="neural-agent")
        self.neural_to_classic_handler = echo.packet_handler(cfo_samps, corr_reps, full_len, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=0.0000,
        	frequency_offset=0.0,
        	epsilon=1.0,
        	taps=(1.0 + 0.0j, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_random_pdu_0 = blocks.random_pdu(2 * packet_len * bits_per_symb, 2 * packet_len * bits_per_symb, chr(0x01), 2)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_gr_complex*1)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_random_pdu_0, 'pdus'), (self.echo_echo_mod_demod_0, 'mod_in'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_echo_mod_demod_0, 'demod_in'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_watchdog_0, 'in'))    
        self.msg_connect((self.echo_echo_mod_demod_0, 'mod_out'), (self.neural_to_classic_handler, 'wrap_in'))    
        self.msg_connect((self.echo_watchdog_0, 'out'), (self.blocks_random_pdu_0, 'generate'))    
        self.msg_connect((self.zeromq_pull_msg_source_0, 'out'), (self.echo_echo_mod_demod_0, 'control'))    
        self.msg_connect((self.zeromq_pull_msg_source_0, 'out'), (self.uhd_usrp_sink_0, 'command'))    
        self.msg_connect((self.zeromq_pull_msg_source_1, 'out'), (self.echo_echo_mod_demod_0, 'control'))    
        self.msg_connect((self.zeromq_pull_msg_source_1, 'out'), (self.uhd_usrp_source_0, 'command'))    
        self.connect((self.channels_channel_model_0, 0), (self.neural_to_classic_handler, 0))    
        self.connect((self.neural_to_classic_handler, 0), (self.uhd_usrp_sink_0, 0))    
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_null_sink_0, 0))    
        self.connect((self.uhd_usrp_source_0, 1), (self.channels_channel_model_0, 0))    

    def get_beta_rrc(self):
        return self.beta_rrc

    def set_beta_rrc(self, beta_rrc):
        self.beta_rrc = beta_rrc

    def get_bits_per_symb(self):
        return self.bits_per_symb

    def set_bits_per_symb(self, bits_per_symb):
        self.bits_per_symb = bits_per_symb
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)
        self.set_spy_length(64 * self.bits_per_symb)

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

    def get_demod_hidden_layers(self):
        return self.demod_hidden_layers

    def set_demod_hidden_layers(self, demod_hidden_layers):
        self.demod_hidden_layers = demod_hidden_layers

    def get_demod_init_weights(self):
        return self.demod_init_weights

    def set_demod_init_weights(self, demod_init_weights):
        self.demod_init_weights = demod_init_weights

    def get_demod_seed(self):
        return self.demod_seed

    def set_demod_seed(self, demod_seed):
        self.demod_seed = demod_seed

    def get_demodtype(self):
        return self.demodtype

    def set_demodtype(self, demodtype):
        self.demodtype = demodtype

    def get_lambda_center(self):
        return self.lambda_center

    def set_lambda_center(self, lambda_center):
        self.lambda_center = lambda_center

    def get_log_interval(self):
        return self.log_interval

    def set_log_interval(self, log_interval):
        self.log_interval = log_interval

    def get_max_amplitude(self):
        return self.max_amplitude

    def set_max_amplitude(self, max_amplitude):
        self.max_amplitude = max_amplitude

    def get_mod_hidden_layers(self):
        return self.mod_hidden_layers

    def set_mod_hidden_layers(self, mod_hidden_layers):
        self.mod_hidden_layers = mod_hidden_layers

    def get_mod_init_weights(self):
        return self.mod_init_weights

    def set_mod_init_weights(self, mod_init_weights):
        self.mod_init_weights = mod_init_weights

    def get_mod_seed(self):
        return self.mod_seed

    def set_mod_seed(self, mod_seed):
        self.mod_seed = mod_seed

    def get_modtype(self):
        return self.modtype

    def set_modtype(self, modtype):
        self.modtype = modtype

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 0)
        	
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 1)
        	

    def get_samps_per_symb(self):
        return self.samps_per_symb

    def set_samps_per_symb(self, samps_per_symb):
        self.samps_per_symb = samps_per_symb

    def get_shared_preamble(self):
        return self.shared_preamble

    def set_shared_preamble(self, shared_preamble):
        self.shared_preamble = shared_preamble

    def get_spy_thresh(self):
        return self.spy_thresh

    def set_spy_thresh(self, spy_thresh):
        self.spy_thresh = spy_thresh

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)
        	
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 1)
        	

    def get_spy_length(self):
        return self.spy_length

    def set_spy_length(self, spy_length):
        self.spy_length = spy_length
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_full_len(self):
        return self.full_len

    def set_full_len(self, full_len):
        self.full_len = full_len

    def get_cfo_freqs(self):
        return self.cfo_freqs

    def set_cfo_freqs(self, cfo_freqs):
        self.cfo_freqs = cfo_freqs

    def get_F_center(self):
        return self.F_center

    def set_F_center(self, F_center):
        self.F_center = F_center
        self.uhd_usrp_sink_0.set_center_freq(self.F_center, 0)
        self.uhd_usrp_sink_0.set_center_freq(self.F_center, 1)
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
        "", "--cfar-thresh", dest="cfar_thresh", type="eng_float", default=eng_notation.num_to_str(50),
        help="Set CFAR Threshold [default=%default]")
    parser.add_option(
        "", "--cfo-samps", dest="cfo_samps", type="intx", default=512,
        help="Set CFO Samples [default=%default]")
    parser.add_option(
        "", "--corr-reps", dest="corr_reps", type="intx", default=2,
        help="Set Correlator Repetitions [default=%default]")
    parser.add_option(
        "", "--demod-init-weights", dest="demod_init_weights", type="string", default="",
        help="Set Demodulator Init Weights [default=%default]")
    parser.add_option(
        "", "--demod-seed", dest="demod_seed", type="intx", default=341133,
        help="Set demod_seed [default=%default]")
    parser.add_option(
        "", "--demodtype", dest="demodtype", type="string", default="neural",
        help="Set Demodulator Type [default=%default]")
    parser.add_option(
        "", "--lambda-center", dest="lambda_center", type="eng_float", default=eng_notation.num_to_str(1),
        help="Set Lambda Center [default=%default]")
    parser.add_option(
        "", "--log-interval", dest="log_interval", type="intx", default=1,
        help="Set Constellation Log Interval [default=%default]")
    parser.add_option(
        "", "--max-amplitude", dest="max_amplitude", type="eng_float", default=eng_notation.num_to_str(0.707),
        help="Set Max Amplitude [default=%default]")
    parser.add_option(
        "", "--mod-init-weights", dest="mod_init_weights", type="string", default="",
        help="Set Modulator Init Weights [default=%default]")
    parser.add_option(
        "", "--mod-seed", dest="mod_seed", type="intx", default=341132,
        help="Set mod_seed [default=%default]")
    parser.add_option(
        "", "--modtype", dest="modtype", type="string", default="neural",
        help="Set Modulator Type [default=%default]")
    parser.add_option(
        "", "--packet-len", dest="packet_len", type="intx", default=256,
        help="Set Packet Length [default=%default]")
    parser.add_option(
        "", "--rx-gain", dest="rx_gain", type="eng_float", default=eng_notation.num_to_str(4),
        help="Set rx_gain [default=%default]")
    parser.add_option(
        "", "--samps-per-symb", dest="samps_per_symb", type="intx", default=2,
        help="Set Samples Per Symbol [default=%default]")
    parser.add_option(
        "", "--shared-preamble", dest="shared_preamble", type="string", default="",
        help="Set Shared Preamble File [default=%default]")
    parser.add_option(
        "", "--spy-thresh", dest="spy_thresh", type="eng_float", default=eng_notation.num_to_str(0.1),
        help="Set Spy Threshold [default=%default]")
    parser.add_option(
        "", "--tx-gain", dest="tx_gain", type="eng_float", default=eng_notation.num_to_str(3),
        help="Set tx_gain [default=%default]")
    return parser


def main(top_block_cls=echo_single_usrp, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    tb = top_block_cls(beta_rrc=options.beta_rrc, bits_per_symb=options.bits_per_symb, cfar_thresh=options.cfar_thresh, cfo_samps=options.cfo_samps, corr_reps=options.corr_reps, demod_init_weights=options.demod_init_weights, demod_seed=options.demod_seed, demodtype=options.demodtype, lambda_center=options.lambda_center, log_interval=options.log_interval, max_amplitude=options.max_amplitude, mod_init_weights=options.mod_init_weights, mod_seed=options.mod_seed, modtype=options.modtype, packet_len=options.packet_len, rx_gain=options.rx_gain, samps_per_symb=options.samps_per_symb, shared_preamble=options.shared_preamble, spy_thresh=options.spy_thresh, tx_gain=options.tx_gain)
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
