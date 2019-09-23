#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Echo Wrapper Neural
# Generated: Wed Apr 17 00:29:53 2019
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt4 import Qt
from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import sip
import sys
import echo


class echo_wrapper_neural(gr.top_block, Qt.QWidget):

    def __init__(self, beta_rrc=0.13, bits_per_symb=2, cfar_thresh=8, cfo_samps=512, corr_reps=1, log_interval=10, packet_len=512, samps_per_symb=2):
        gr.top_block.__init__(self, "Echo Wrapper Neural")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Echo Wrapper Neural")
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

        self.settings = Qt.QSettings("GNU Radio", "echo_wrapper_neural")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

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

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1000000
        self.cfo_freqs = cfo_freqs = 11./256, 43./256, 97./256
        self.body = body = [1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0]
        self.N0 = N0 = 0.1

        ##################################################
        # Blocks
        ##################################################
        self.echo_triggered_vector_source_b_0 = echo.triggered_vector_source_b((body * 2), "", 1, "pkt")
        self.echo_preamble_insert_1_0 = echo.preamble_insert((body), 1, 0, 1)
        self.echo_preamble_insert_1 = echo.preamble_insert((body), 0, 1, 0)
        self.echo_preamble_insert_0_0 = echo.preamble_insert((body), 0, 1, 0)
        self.echo_preamble_insert_0 = echo.preamble_insert((body), 1, 0, 1)
        self.echo_packet_length_check_0_0 = echo.packet_length_check(packet_len / bits_per_symb * 2)
        self.echo_packet_length_check_0 = echo.packet_length_check(packet_len / bits_per_symb * 2)
        self.echo_modulator_neural_0 = echo.modulator_neural(12348907843, (50, ), bits_per_symb, (body), log_interval, "")
        self.echo_modulator_classic_0 = echo.modulator_classic(bits_per_symb, (body), log_interval)
        self.echo_demodulator_neural_0 = echo.demodulator_neural(134214, (50, ), bits_per_symb, (body), log_interval, "")
        self.echo_demodulator_classic_0 = echo.demodulator_classic(bits_per_symb, 1024, preamble=(body), log_ber_interval=log_interval)
        self.qtgui_const_sink_x_0_0_0 = qtgui.const_sink_c(
        	2048, #size
        	"Channel IQ Scatter", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_0.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0_0.enable_grid(True)
        
        if not True:
          self.qtgui_const_sink_x_0_0_0.disable_legend()
        
        labels = ["", "", "", "", "",
                  "", "", "", "", ""]
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
                self.qtgui_const_sink_x_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_0.set_line_alpha(i, alphas[i])
        
        self._qtgui_const_sink_x_0_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_0_win)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
        	packet_len / bits_per_symb * 2, #size
        	"Detected Body Constellation", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(True)
        
        if not True:
          self.qtgui_const_sink_x_0_0.disable_legend()
        
        labels = ["", "", "", "", "",
                  "", "", "", "", ""]
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
        self.neural_to_classic_handler = echo.packet_handler(cfo_samps, corr_reps, packet_len / samps_per_symb * 2, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.classic_to_neural_handler = echo.packet_handler(cfo_samps, corr_reps, 2 * packet_len / samps_per_symb, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.channels_channel_model_0_0 = channels.channel_model(
        	noise_voltage=N0,
        	frequency_offset=0.000,
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
        self.blocks_tagged_stream_to_pdu_0_0 = blocks.tagged_stream_to_pdu(blocks.byte_t, "pkt")
        self.blocks_stream_mux_1 = blocks.stream_mux(gr.sizeof_gr_complex*1, (1152 * 2, 5000))
        self.blocks_stream_mux_0 = blocks.stream_mux(gr.sizeof_gr_complex*1, (1152 * 2, 5000))
        self.blocks_pdu_to_tagged_stream_0_1 = blocks.pdu_to_tagged_stream(blocks.complex_t, "pkt")
        self.analog_const_source_x_0_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0)
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 0)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_tagged_stream_to_pdu_0_0, 'pdus'), (self.echo_modulator_neural_0, 'bits'))    
        self.msg_connect((self.classic_to_neural_handler, 'unwrap_out'), (self.echo_packet_length_check_0, 'in'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.blocks_pdu_to_tagged_stream_0_1, 'pdus'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_packet_length_check_0_0, 'in'))    
        self.msg_connect((self.echo_demodulator_classic_0, 'bits'), (self.echo_preamble_insert_0, 'in'))    
        self.msg_connect((self.echo_demodulator_classic_0, 'bits'), (self.echo_preamble_insert_0_0, 'in'))    
        self.msg_connect((self.echo_demodulator_neural_0, 'bits'), (self.echo_preamble_insert_1, 'in'))    
        self.msg_connect((self.echo_demodulator_neural_0, 'bits'), (self.echo_preamble_insert_1_0, 'in'))    
        self.msg_connect((self.echo_modulator_classic_0, 'symbols'), (self.classic_to_neural_handler, 'wrap_in'))    
        self.msg_connect((self.echo_modulator_neural_0, 'symbols'), (self.neural_to_classic_handler, 'wrap_in'))    
        self.msg_connect((self.echo_packet_length_check_0, 'validated'), (self.echo_demodulator_neural_0, 'symbols'))    
        self.msg_connect((self.echo_packet_length_check_0, 'failed'), (self.echo_triggered_vector_source_b_0, 'trigger'))    
        self.msg_connect((self.echo_packet_length_check_0_0, 'validated'), (self.echo_demodulator_classic_0, 'symbols'))    
        self.msg_connect((self.echo_packet_length_check_0_0, 'failed'), (self.echo_triggered_vector_source_b_0, 'trigger'))    
        self.msg_connect((self.echo_preamble_insert_0, 'out'), (self.echo_modulator_classic_0, 'bits'))    
        self.msg_connect((self.echo_preamble_insert_0_0, 'out'), (self.echo_modulator_classic_0, 'update'))    
        self.msg_connect((self.echo_preamble_insert_1, 'out'), (self.echo_modulator_neural_0, 'feedback'))    
        self.msg_connect((self.echo_preamble_insert_1_0, 'out'), (self.echo_modulator_neural_0, 'bits'))    
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_stream_mux_1, 1))    
        self.connect((self.analog_const_source_x_0_0, 0), (self.blocks_stream_mux_0, 1))    
        self.connect((self.blocks_pdu_to_tagged_stream_0_1, 0), (self.qtgui_const_sink_x_0_0, 0))    
        self.connect((self.blocks_stream_mux_0, 0), (self.channels_channel_model_0, 0))    
        self.connect((self.blocks_stream_mux_1, 0), (self.channels_channel_model_0_0, 0))    
        self.connect((self.blocks_throttle_0, 0), (self.neural_to_classic_handler, 0))    
        self.connect((self.channels_channel_model_0, 0), (self.blocks_throttle_0, 0))    
        self.connect((self.channels_channel_model_0, 0), (self.qtgui_const_sink_x_0_0_0, 0))    
        self.connect((self.channels_channel_model_0_0, 0), (self.classic_to_neural_handler, 0))    
        self.connect((self.classic_to_neural_handler, 0), (self.blocks_stream_mux_1, 0))    
        self.connect((self.neural_to_classic_handler, 0), (self.blocks_stream_mux_0, 0))    
        self.connect((self.echo_triggered_vector_source_b_0, 0), (self.blocks_tagged_stream_to_pdu_0_0, 0))    

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "echo_wrapper_neural")
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

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

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
        self.channels_channel_model_0.set_noise_voltage(self.N0)
        self.channels_channel_model_0_0.set_noise_voltage(self.N0)


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
        "", "--corr-reps", dest="corr_reps", type="intx", default=1,
        help="Set Correlator Repetitions [default=%default]")
    parser.add_option(
        "", "--log-interval", dest="log_interval", type="intx", default=10,
        help="Set Constellation Log Interval [default=%default]")
    parser.add_option(
        "", "--packet-len", dest="packet_len", type="intx", default=512,
        help="Set Packet Length [default=%default]")
    parser.add_option(
        "", "--samps-per-symb", dest="samps_per_symb", type="intx", default=2,
        help="Set Samples Per Symbol [default=%default]")
    return parser


def main(top_block_cls=echo_wrapper_neural, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(beta_rrc=options.beta_rrc, bits_per_symb=options.bits_per_symb, cfar_thresh=options.cfar_thresh, cfo_samps=options.cfo_samps, corr_reps=options.corr_reps, log_interval=options.log_interval, packet_len=options.packet_len, samps_per_symb=options.samps_per_symb)
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
