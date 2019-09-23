#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Echo Wrapper Neural Spy
# Generated: Fri Jun 14 14:05:32 2019
##################################################

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt5 import Qt
from PyQt5 import Qt, QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio import zeromq
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import pmt
import sip
import sys
import echo
from gnuradio import qtgui


class echo_wrapper_neural_spy(gr.top_block, Qt.QWidget):

    def __init__(self, beta_rrc=0.13, bits_per_symb=2, cfar_thresh=8, cfo_samps=512, corr_reps=2, demod_hidden_layers=[50], demodtype='neural', log_interval=1, mod_hidden_layers=[100], modtype='neural', packet_len=256, samps_per_symb=2, spy_thresh=0.1):
        gr.top_block.__init__(self, "Echo Wrapper Neural Spy")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Echo Wrapper Neural Spy")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "echo_wrapper_neural_spy")

        if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
            self.restoreGeometry(self.settings.value("geometry").toByteArray())
        else:
            self.restoreGeometry(self.settings.value("geometry", type=QtCore.QByteArray))

        ##################################################
        # Parameters
        ##################################################
        self.beta_rrc = beta_rrc
        self.bits_per_symb = bits_per_symb
        self.cfar_thresh = cfar_thresh
        self.cfo_samps = cfo_samps
        self.corr_reps = corr_reps
        self.demod_hidden_layers = demod_hidden_layers
        self.demodtype = demodtype
        self.log_interval = log_interval
        self.mod_hidden_layers = mod_hidden_layers
        self.modtype = modtype
        self.packet_len = packet_len
        self.samps_per_symb = samps_per_symb
        self.spy_thresh = spy_thresh

        ##################################################
        # Variables
        ##################################################
        self.spy_length = spy_length = 64 * bits_per_symb
        self.full_len = full_len = 2 * packet_len + (spy_length + 96) / bits_per_symb
        self.samp_rate = samp_rate = 500000
        self.nsamps = nsamps = samps_per_symb * (256 * corr_reps * 2 + 64 * 2 + full_len)
        self.cfo_freqs = cfo_freqs = 11./256, 43./256, 97./256
        self.body = body = [1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0]
        self.N0 = N0 = 0.1

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_pull_msg_source_1 = zeromq.pull_msg_source('tcp://127.0.0.1:5556', 100)
        self.zeromq_pull_msg_source_0 = zeromq.pull_msg_source('tcp://127.0.0.1:5555', 100)
        self.echo_watchdog_0 = echo.watchdog(1)
        self.echo_echo_mod_demod_0_0 = echo.echo_mod_demod(packet_len * bits_per_symb, (), bits_per_symb,
              'classic', 'classic', 128, 129,
              (mod_hidden_layers), (demod_hidden_layers),
              '', '',
              log_interval, spy_length, 0.1, 'classic-agent')
        self.echo_echo_mod_demod_0 = echo.echo_mod_demod(packet_len * bits_per_symb, (), bits_per_symb,
              modtype, demodtype, 128, 129,
              (mod_hidden_layers), (demod_hidden_layers),
              '', '',
              log_interval, spy_length, 0.1, 'neural-agent')
        self.qtgui_const_sink_x_0_0_1 = qtgui.const_sink_c(
        	full_len, #size
        	"Detected Body Constellation", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_1.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0_1.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_1.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_1.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0_1.enable_grid(True)
        self.qtgui_const_sink_x_0_0_1.enable_axis_labels(True)

        if not True:
          self.qtgui_const_sink_x_0_0_1.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
                  "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_1.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_1.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_1.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_1.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_1.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_1.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_1_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_1.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_1_win)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
        	full_len, #size
        	"Detected Body Constellation", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(True)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)

        if not True:
          self.qtgui_const_sink_x_0_0.disable_legend()

        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
                  "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in xrange(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_win)
        self.neural_to_classic_handler = echo.packet_handler(cfo_samps, corr_reps, full_len, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.classic_to_neural_handler = echo.packet_handler(cfo_samps, corr_reps, full_len, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.channels_channel_model_0_0 = channels.channel_model(
        	noise_voltage=N0,
        	frequency_offset=0.0000,
        	epsilon=1,
        	taps=(0.714+0.714j, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=N0,
        	frequency_offset=0.0000,
        	epsilon=1,
        	taps=(0.714 + 0.714j, ),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_mux_1 = blocks.stream_mux(gr.sizeof_gr_complex*1, (nsamps, 20000))
        self.blocks_stream_mux_0 = blocks.stream_mux(gr.sizeof_gr_complex*1, (nsamps, 20000))
        self.blocks_random_pdu_0 = blocks.random_pdu(2 * packet_len * bits_per_symb, 2 * packet_len * bits_per_symb, chr(0x01), 2)
        self.blocks_pdu_to_tagged_stream_0_1_0 = blocks.pdu_to_tagged_stream(blocks.complex_t, 'pkt')
        self.blocks_pdu_to_tagged_stream_0_1 = blocks.pdu_to_tagged_stream(blocks.complex_t, 'pkt')
        self.analog_const_source_x_0_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_random_pdu_0, 'pdus'), (self.echo_echo_mod_demod_0, 'mod_in'))
        self.msg_connect((self.classic_to_neural_handler, 'unwrap_out'), (self.blocks_pdu_to_tagged_stream_0_1_0, 'pdus'))
        self.msg_connect((self.classic_to_neural_handler, 'unwrap_out'), (self.echo_echo_mod_demod_0_0, 'demod_in'))
        self.msg_connect((self.classic_to_neural_handler, 'unwrap_out'), (self.echo_watchdog_0, 'in'))
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.blocks_pdu_to_tagged_stream_0_1, 'pdus'))
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_echo_mod_demod_0, 'demod_in'))
        self.msg_connect((self.echo_echo_mod_demod_0, 'mod_out'), (self.neural_to_classic_handler, 'wrap_in'))
        self.msg_connect((self.echo_echo_mod_demod_0_0, 'mod_out'), (self.classic_to_neural_handler, 'wrap_in'))
        self.msg_connect((self.echo_watchdog_0, 'out'), (self.blocks_random_pdu_0, 'generate'))
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_stream_mux_1, 1))
        self.connect((self.analog_const_source_x_0_0, 0), (self.blocks_stream_mux_0, 1))
        self.connect((self.blocks_pdu_to_tagged_stream_0_1, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.blocks_pdu_to_tagged_stream_0_1_0, 0), (self.qtgui_const_sink_x_0_0_1, 0))
        self.connect((self.blocks_stream_mux_0, 0), (self.channels_channel_model_0_0, 0))
        self.connect((self.blocks_stream_mux_1, 0), (self.channels_channel_model_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.neural_to_classic_handler, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.classic_to_neural_handler, 0))
        self.connect((self.classic_to_neural_handler, 0), (self.blocks_stream_mux_1, 0))
        self.connect((self.neural_to_classic_handler, 0), (self.blocks_stream_mux_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "echo_wrapper_neural_spy")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_beta_rrc(self):
        return self.beta_rrc

    def set_beta_rrc(self, beta_rrc):
        self.beta_rrc = beta_rrc

    def get_bits_per_symb(self):
        return self.bits_per_symb

    def set_bits_per_symb(self, bits_per_symb):
        self.bits_per_symb = bits_per_symb
        self.set_spy_length(64 * self.bits_per_symb)
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)

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
        self.set_nsamps(self.samps_per_symb * (256 * self.corr_reps * 2 + 64 * 2 + self.full_len))

    def get_demod_hidden_layers(self):
        return self.demod_hidden_layers

    def set_demod_hidden_layers(self, demod_hidden_layers):
        self.demod_hidden_layers = demod_hidden_layers

    def get_demodtype(self):
        return self.demodtype

    def set_demodtype(self, demodtype):
        self.demodtype = demodtype

    def get_log_interval(self):
        return self.log_interval

    def set_log_interval(self, log_interval):
        self.log_interval = log_interval

    def get_mod_hidden_layers(self):
        return self.mod_hidden_layers

    def set_mod_hidden_layers(self, mod_hidden_layers):
        self.mod_hidden_layers = mod_hidden_layers

    def get_modtype(self):
        return self.modtype

    def set_modtype(self, modtype):
        self.modtype = modtype

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)

    def get_samps_per_symb(self):
        return self.samps_per_symb

    def set_samps_per_symb(self, samps_per_symb):
        self.samps_per_symb = samps_per_symb
        self.set_nsamps(self.samps_per_symb * (256 * self.corr_reps * 2 + 64 * 2 + self.full_len))

    def get_spy_thresh(self):
        return self.spy_thresh

    def set_spy_thresh(self, spy_thresh):
        self.spy_thresh = spy_thresh

    def get_spy_length(self):
        return self.spy_length

    def set_spy_length(self, spy_length):
        self.spy_length = spy_length
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)

    def get_full_len(self):
        return self.full_len

    def set_full_len(self, full_len):
        self.full_len = full_len
        self.set_nsamps(self.samps_per_symb * (256 * self.corr_reps * 2 + 64 * 2 + self.full_len))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_nsamps(self):
        return self.nsamps

    def set_nsamps(self, nsamps):
        self.nsamps = nsamps

    def get_cfo_freqs(self):
        return self.cfo_freqs

    def set_cfo_freqs(self, cfo_freqs):
        self.cfo_freqs = cfo_freqs

    def get_body(self):
        return self.body

    def set_body(self, body):
        self.body = body

    def get_N0(self):
        return self.N0

    def set_N0(self, N0):
        self.N0 = N0
        self.channels_channel_model_0_0.set_noise_voltage(self.N0)
        self.channels_channel_model_0.set_noise_voltage(self.N0)


def argument_parser():
    parser = OptionParser(usage="%prog: [options]", option_class=eng_option)
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
        "", "--demodtype", dest="demodtype", type="string", default='neural',
        help="Set Demodulator Type [default=%default]")
    parser.add_option(
        "", "--log-interval", dest="log_interval", type="intx", default=1,
        help="Set Constellation Log Interval [default=%default]")
    parser.add_option(
        "", "--modtype", dest="modtype", type="string", default='neural',
        help="Set Modulator Type [default=%default]")
    parser.add_option(
        "", "--packet-len", dest="packet_len", type="intx", default=256,
        help="Set Packet Length [default=%default]")
    parser.add_option(
        "", "--samps-per-symb", dest="samps_per_symb", type="intx", default=2,
        help="Set Samples Per Symbol [default=%default]")
    parser.add_option(
        "", "--spy-thresh", dest="spy_thresh", type="eng_float", default=eng_notation.num_to_str(0.1),
        help="Set Spy Threshold [default=%default]")
    return parser


def main(top_block_cls=echo_wrapper_neural_spy, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print "Error: failed to enable real-time scheduling."

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(beta_rrc=options.beta_rrc, bits_per_symb=options.bits_per_symb, cfar_thresh=options.cfar_thresh, cfo_samps=options.cfo_samps, corr_reps=options.corr_reps, demodtype=options.demodtype, log_interval=options.log_interval, modtype=options.modtype, packet_len=options.packet_len, samps_per_symb=options.samps_per_symb, spy_thresh=options.spy_thresh)
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
