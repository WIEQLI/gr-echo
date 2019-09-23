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

########################################################################
# Include python install macros
########################################################################
include(GrPython)
if(NOT PYTHONINTERP_FOUND)
    return()
endif()

########################################################################
# Install python sources
########################################################################
GR_PYTHON_INSTALL(
    FILES
    __init__.py
    modulator_classic.py
    demodulator_classic.py
    demodulator_neural.py
    modulator_neural.py
    demodulator_neural.py
    triggered_vector_source_b.py
    EchoPacketWrapper.py
    DSPUtil.py
    peakdetect.py
    echo_packet_wrapper.py
    echo_packet_unwrapper.py
    echo_packet_detect.py
    rand_zeros_extend_c.py
    packet_length_check.py
    preamble_insert.py
    packet_handler.py
    add_message_to_stream_async.py
    watchdog.py
    demodulator_classic_spy.py
    demodulator_neural_spy.py
    modulator_classic_spy.py
    modulator_neural_spy.py
    echo_mod_demod.py DESTINATION ${GR_PYTHON_DIR}/echo
)
# Install torch_echo module implementing learning modulators
FILE(GLOB PYECHOECHO torch_echo/*.py)
FILE(GLOB PYECHOECHO_MOD torch_echo/modulators/*.py)
FILE(GLOB PYECHOECHO_DEMOD torch_echo/demodulators/*.py)
FILE(GLOB PYECHOECHO_UTIL torch_echo/utils/*.py)
INSTALL(FILES ${PYECHOECHO} DESTINATION ${GR_PYTHON_DIR}/echo/torch_echo)
INSTALL(FILES ${PYECHOECHO_MOD} DESTINATION ${GR_PYTHON_DIR}/echo/torch_echo/modulators)
INSTALL(FILES ${PYECHOECHO_DEMOD} DESTINATION ${GR_PYTHON_DIR}/echo/torch_echo/demodulators)
INSTALL(FILES ${PYECHOECHO_UTIL} DESTINATION ${GR_PYTHON_DIR}/echo/torch_echo/utils)
# Also copy the files to the build directory so that tests will work
# Copy full directory
macro(resource_dirs dirs)
    foreach(dir ${dirs})
        # Replace / at the end of the path (copy dir content VS copy dir)
        string(REGEX REPLACE "/+$" "" dirclean "${dir}")
        message(STATUS "Copying resource ${dirclean}")
        file(COPY ${dirclean} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
endmacro()
resource_dirs("torch_echo/")

########################################################################
# Handle the unit tests
########################################################################
include(GrTest)

set(GR_TEST_TARGET_DEPS gnuradio-echo)
set(GR_TEST_PYTHON_DIRS ${CMAKE_BINARY_DIR}/swig)
GR_ADD_TEST(qa_modulator_classic ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_modulator_classic.py)
GR_ADD_TEST(qa_demodulator_classic ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_demodulator_classic.py)
GR_ADD_TEST(qa_demodulator_neural ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_demodulator_neural.py)
GR_ADD_TEST(qa_modulator_neural ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_modulator_neural.py)
GR_ADD_TEST(qa_demodulator_neural ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_demodulator_neural.py)
GR_ADD_TEST(qa_triggered_vector_source_b ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_triggered_vector_source_b.py)
GR_ADD_TEST(qa_echo_packet_wrapper ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_echo_packet_wrapper.py)
GR_ADD_TEST(qa_echo_packet_unwrapper ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_echo_packet_unwrapper.py)
GR_ADD_TEST(qa_echo_packet_detect ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_echo_packet_detect.py)
GR_ADD_TEST(qa_add_message_to_stream_async ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/qa_add_message_to_stream_async.py)
