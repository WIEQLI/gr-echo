#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Echo Neural Usrp Loop
# Generated: Wed Jul  3 22:33:18 2019
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
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio import uhd
from gnuradio import zeromq
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import pmt
import sip
import sys
import echo
import time


class echo_neural_usrp_loop(gr.top_block, Qt.QWidget):

    def __init__(self, beta_rrc=0.13, bits_per_symb=2, cfar_thresh=15, cfo_samps=512, corr_reps=2, demod_hidden_layers=[50], demodtype="classic", log_interval=1, max_amplitude=0.707, mod_hidden_layers=[50], modtype="classic", packet_len=256 , samps_per_symb=2, spy_thresh=0.1):
        gr.top_block.__init__(self, "Echo Neural Usrp Loop")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Echo Neural Usrp Loop")
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

        self.settings = Qt.QSettings("GNU Radio", "echo_neural_usrp_loop")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

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
        self.max_amplitude = max_amplitude
        self.mod_hidden_layers = mod_hidden_layers
        self.modtype = modtype
        self.packet_len = packet_len
        self.samps_per_symb = samps_per_symb
        self.spy_thresh = spy_thresh

        ##################################################
        # Variables
        ##################################################
        self.spy_length = spy_length = 64 * bits_per_symb
        self.tx_gain = tx_gain = 5
        self.samp_rate = samp_rate = 500000
        self.rx_gain = rx_gain = 40
        self.full_len = full_len = 2 * packet_len + (spy_length + 96) / bits_per_symb
        self.cfo_freqs = cfo_freqs = 11./256, 43./256, 97./256
        self.body = body = [1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0]
        self.F_center = F_center = 1000000000

        ##################################################
        # Blocks
        ##################################################
        self.zeromq_pull_msg_source_0_0 = zeromq.pull_msg_source("tcp://127.0.0.1:5556", 100, True)
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
        		channels=range(2),
        	),
        	"packet_len",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(F_center, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_center_freq(F_center, 1)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 1)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 1)
        self.echo_watchdog_0_0 = echo.watchdog(1.5)
        self.echo_watchdog_0 = echo.watchdog(1)
        self.echo_echo_mod_demod_0_0 = echo.echo_mod_demod(packet_len * bits_per_symb, (), bits_per_symb, 
              "classic", "classic", 128, 129, 
              (mod_hidden_layers), (demod_hidden_layers), 
              "", "",
              log_interval, spy_length, spy_thresh, 
              max_amplitude, "neural-agent")
        self.echo_echo_mod_demod_0 = echo.echo_mod_demod(packet_len * bits_per_symb, (), bits_per_symb, 
              modtype, demodtype, 1128, 1129, 
              (mod_hidden_layers), (demod_hidden_layers), 
              "", "",
              log_interval, spy_length, spy_thresh, 
              max_amplitude, "neural-agent")
        self.qtgui_const_sink_x_0_0_0_0 = qtgui.const_sink_c(
        	full_len, #size
        	"Neural Mod Output Scatter", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_0_0.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0_0_0.set_y_axis(-1.5, 1.5)
        self.qtgui_const_sink_x_0_0_0_0.set_x_axis(-1.5, 1.5)
        self.qtgui_const_sink_x_0_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0_0_0.enable_grid(True)
        
        if not True:
          self.qtgui_const_sink_x_0_0_0_0.disable_legend()
        
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
                self.qtgui_const_sink_x_0_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0_0_0.set_line_alpha(i, alphas[i])
        
        self._qtgui_const_sink_x_0_0_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0_0_0.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_0_0_0_win, 0,0,1,1)
        self.qtgui_const_sink_x_0_0_0 = qtgui.const_sink_c(
        	full_len, #size
        	"Neural Agent Rx Scatter", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0_0.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0_0.set_y_axis(-1.5, 1.5)
        self.qtgui_const_sink_x_0_0_0.set_x_axis(-1.5, 1.5)
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
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_0_0_win, 1,1,1,1)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
        	full_len, #size
        	"Classic Agent Rx Scatter", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.0)
        self.qtgui_const_sink_x_0_0.set_y_axis(-1.5, 1.5)
        self.qtgui_const_sink_x_0_0.set_x_axis(-1.5, 1.5)
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
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_0_win, 0,1,1,1)
        self.neural_to_classic_handler_0 = echo.packet_handler(cfo_samps, corr_reps, full_len, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.neural_to_classic_handler = echo.packet_handler(cfo_samps, corr_reps, full_len, cfar_thresh, samps_per_symb, beta_rrc, (cfo_freqs))
        self.blocks_random_pdu_0_0 = blocks.random_pdu(2 * packet_len * bits_per_symb , 2 * packet_len * bits_per_symb, chr(0x01), 2)
        self.blocks_random_pdu_0 = blocks.random_pdu(2 * packet_len * bits_per_symb, 2 * packet_len * bits_per_symb, chr(0x01), 2)
        self.blocks_pdu_to_tagged_stream_0_1_0_0 = blocks.pdu_to_tagged_stream(blocks.complex_t, "pkt")
        self.blocks_pdu_to_tagged_stream_0_1_0 = blocks.pdu_to_tagged_stream(blocks.complex_t, "pkt")
        self.blocks_pdu_to_tagged_stream_0_1 = blocks.pdu_to_tagged_stream(blocks.complex_t, "pkt")
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, "loop-iq.bin", False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_random_pdu_0, 'pdus'), (self.echo_echo_mod_demod_0, 'mod_in'))    
        self.msg_connect((self.blocks_random_pdu_0_0, 'pdus'), (self.echo_echo_mod_demod_0_0, 'mod_in'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.blocks_pdu_to_tagged_stream_0_1_0, 'pdus'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_echo_mod_demod_0, 'demod_in'))    
        self.msg_connect((self.neural_to_classic_handler, 'unwrap_out'), (self.echo_watchdog_0, 'in'))    
        self.msg_connect((self.neural_to_classic_handler_0, 'unwrap_out'), (self.blocks_pdu_to_tagged_stream_0_1, 'pdus'))    
        self.msg_connect((self.neural_to_classic_handler_0, 'unwrap_out'), (self.echo_echo_mod_demod_0_0, 'demod_in'))    
        self.msg_connect((self.neural_to_classic_handler_0, 'unwrap_out'), (self.echo_watchdog_0_0, 'in'))    
        self.msg_connect((self.echo_echo_mod_demod_0, 'mod_out'), (self.blocks_pdu_to_tagged_stream_0_1_0_0, 'pdus'))    
        self.msg_connect((self.echo_echo_mod_demod_0, 'mod_out'), (self.neural_to_classic_handler, 'wrap_in'))    
        self.msg_connect((self.echo_echo_mod_demod_0_0, 'mod_out'), (self.neural_to_classic_handler_0, 'wrap_in'))    
        self.msg_connect((self.echo_watchdog_0, 'out'), (self.blocks_random_pdu_0, 'generate'))    
        self.msg_connect((self.echo_watchdog_0_0, 'out'), (self.blocks_random_pdu_0_0, 'generate'))    
        self.msg_connect((self.zeromq_pull_msg_source_0, 'out'), (self.echo_echo_mod_demod_0, 'control'))    
        self.msg_connect((self.zeromq_pull_msg_source_0, 'out'), (self.echo_echo_mod_demod_0_0, 'control'))    
        self.msg_connect((self.zeromq_pull_msg_source_0, 'out'), (self.uhd_usrp_sink_0, 'command'))    
        self.msg_connect((self.zeromq_pull_msg_source_0_0, 'out'), (self.echo_echo_mod_demod_0, 'control'))    
        self.msg_connect((self.zeromq_pull_msg_source_0_0, 'out'), (self.echo_echo_mod_demod_0_0, 'control'))    
        self.msg_connect((self.zeromq_pull_msg_source_0_0, 'out'), (self.uhd_usrp_source_0, 'command'))    
        self.connect((self.blocks_pdu_to_tagged_stream_0_1, 0), (self.qtgui_const_sink_x_0_0, 0))    
        self.connect((self.blocks_pdu_to_tagged_stream_0_1_0, 0), (self.qtgui_const_sink_x_0_0_0, 0))    
        self.connect((self.blocks_pdu_to_tagged_stream_0_1_0_0, 0), (self.qtgui_const_sink_x_0_0_0_0, 0))    
        self.connect((self.neural_to_classic_handler, 0), (self.uhd_usrp_sink_0, 0))    
        self.connect((self.neural_to_classic_handler_0, 0), (self.uhd_usrp_sink_0, 1))    
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0, 0))    
        self.connect((self.uhd_usrp_source_0, 0), (self.neural_to_classic_handler, 0))    
        self.connect((self.uhd_usrp_source_0, 1), (self.neural_to_classic_handler_0, 0))    

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "echo_neural_usrp_loop")
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

    def get_demodtype(self):
        return self.demodtype

    def set_demodtype(self, demodtype):
        self.demodtype = demodtype

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

    def get_spy_thresh(self):
        return self.spy_thresh

    def set_spy_thresh(self, spy_thresh):
        self.spy_thresh = spy_thresh

    def get_spy_length(self):
        return self.spy_length

    def set_spy_length(self, spy_length):
        self.spy_length = spy_length
        self.set_full_len(2 * self.packet_len + (self.spy_length + 96) / self.bits_per_symb)

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)
        	
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 1)
        	

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
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 0)
        	
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 1)
        	

    def get_full_len(self):
        return self.full_len

    def set_full_len(self, full_len):
        self.full_len = full_len

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
        "", "--cfar-thresh", dest="cfar_thresh", type="eng_float", default=eng_notation.num_to_str(15),
        help="Set CFAR Threshold [default=%default]")
    parser.add_option(
        "", "--cfo-samps", dest="cfo_samps", type="intx", default=512,
        help="Set CFO Samples [default=%default]")
    parser.add_option(
        "", "--corr-reps", dest="corr_reps", type="intx", default=2,
        help="Set Correlator Repetitions [default=%default]")
    parser.add_option(
        "", "--demodtype", dest="demodtype", type="string", default="classic",
        help="Set Demodulator Type [default=%default]")
    parser.add_option(
        "", "--log-interval", dest="log_interval", type="intx", default=1,
        help="Set Constellation Log Interval [default=%default]")
    parser.add_option(
        "", "--max-amplitude", dest="max_amplitude", type="eng_float", default=eng_notation.num_to_str(0.707),
        help="Set Max Amplitude [default=%default]")
    parser.add_option(
        "", "--modtype", dest="modtype", type="string", default="classic",
        help="Set Modulator Type [default=%default]")
    parser.add_option(
        "", "--packet-len", dest="packet_len", type="intx", default=256 ,
        help="Set Packet Length [default=%default]")
    parser.add_option(
        "", "--samps-per-symb", dest="samps_per_symb", type="intx", default=2,
        help="Set Samples Per Symbol [default=%default]")
    parser.add_option(
        "", "--spy-thresh", dest="spy_thresh", type="eng_float", default=eng_notation.num_to_str(0.1),
        help="Set Spy Threshold [default=%default]")
    return parser


def main(top_block_cls=echo_neural_usrp_loop, options=None):
    if options is None:
        options, _ = argument_parser().parse_args()

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(beta_rrc=options.beta_rrc, bits_per_symb=options.bits_per_symb, cfar_thresh=options.cfar_thresh, cfo_samps=options.cfo_samps, corr_reps=options.corr_reps, demodtype=options.demodtype, log_interval=options.log_interval, max_amplitude=options.max_amplitude, modtype=options.modtype, packet_len=options.packet_len, samps_per_symb=options.samps_per_symb, spy_thresh=options.spy_thresh)
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
