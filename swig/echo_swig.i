/* -*- c++ -*- */

#define ECHO_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "echo_swig_doc.i"

%{
#include "echo/pdu_complex_to_ettus_tagged_stream.h"
%}


%include "echo/pdu_complex_to_ettus_tagged_stream.h"
GR_SWIG_BLOCK_MAGIC2(echo, pdu_complex_to_ettus_tagged_stream);
