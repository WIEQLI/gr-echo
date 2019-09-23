/* -*- c++ -*- */
/* 
 * Copyright 2018 <+YOU OR YOUR COMPANY+>.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "pdu_complex_to_ettus_tagged_stream_impl.h"

namespace gr {
  namespace echo {

    pdu_complex_to_ettus_tagged_stream::sptr
    pdu_complex_to_ettus_tagged_stream::make(const std::string& length_tag_name, const std::string& sob_tag_name, const std::string& eob_tag_name, int timeout_ms)
    {
      return gnuradio::get_initial_sptr
        (new pdu_complex_to_ettus_tagged_stream_impl(length_tag_name, sob_tag_name, eob_tag_name, timeout_ms));
    }

    /*
     * The private constructor
     */
    pdu_complex_to_ettus_tagged_stream_impl::pdu_complex_to_ettus_tagged_stream_impl(const std::string& length_tag_name, const std::string& sob_tag_name, const std::string& eob_tag_name, int timeout_ms)
      : gr::tagged_stream_block("pdu_complex_to_ettus_tagged_stream",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(1, 1, sizeof(gr_complex)), length_tag_name),
        d_itemsize(gr::blocks::pdu::itemsize(gr::blocks::pdu::complex_t)),
        d_type(gr::blocks::pdu::complex_t),
        d_curr_len(0),
        d_sob_tag_name(sob_tag_name),
        d_eob_tag_name(eob_tag_name),
        d_timeout_ms(timeout_ms)
      {
        message_port_register_in(PDU_PORT_ID);
      }

    /*
     * Our virtual destructor.
     */
    pdu_complex_to_ettus_tagged_stream_impl::~pdu_complex_to_ettus_tagged_stream_impl()
    {
    }

    int
    pdu_complex_to_ettus_tagged_stream_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {   
      if (d_curr_len == 0) {
          /* FIXME: This blocking call is far from ideal but is the best we
       *        can do at the moment
       */
        pmt::pmt_t msg(delete_head_blocking(PDU_PORT_ID, d_timeout_ms)); //The default timeout is 100.  Make this configurable
        if (msg.get() == NULL) {
          return 0;
        }

        if (!pmt::is_pair(msg))
          throw std::runtime_error("received a malformed pdu message");

        d_curr_meta = pmt::car(msg);
        d_curr_vect = pmt::cdr(msg);
        // do not assume the length of  PMT is in items (e.g.: from socket_pdu)        
        d_curr_len = pmt::blob_length(d_curr_vect)/d_itemsize;
      }

      return d_curr_len;
          //(d_curr_len % 4 == 0) ? d_curr_len/4 : d_curr_len/4+1; //Since we are outputing int32_t with uint8_t.  Round up and potentially pad later
    }

    int
    pdu_complex_to_ettus_tagged_stream_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      gr_complex *out = (gr_complex *) output_items[0];
      
      if (d_curr_len == 0) {
        return 0;
      } 

      // work() should only be called if the current PDU fits entirely
      // into the output buffer.
      assert(noutput_items >= d_curr_len);

      // Copy vector output
      size_t io(0);
      const gr_complex *ptr = (const gr_complex*) uniform_vector_elements(d_curr_vect, io);
      memcpy(out, ptr, d_curr_len * d_itemsize);

      // Copy tags
      if (!pmt::eq(d_curr_meta, pmt::PMT_NIL)) {
        pmt::pmt_t klist(pmt::dict_keys(d_curr_meta));
        for (size_t i = 0; i < pmt::length(klist); i++) {
          pmt::pmt_t k(pmt::nth(i, klist));
          pmt::pmt_t v(pmt::dict_ref(d_curr_meta, k, pmt::PMT_NIL));
          add_item_tag(0, nitems_written(0), k, v, alias_pmt());
        }
      }

      // Add Ettus tags
      add_item_tag(0, nitems_written(0), pmt::string_to_symbol(d_sob_tag_name), pmt::PMT_T, alias_pmt());
      add_item_tag(0, nitems_written(0) + d_curr_len - 1, pmt::string_to_symbol(d_eob_tag_name), pmt::PMT_T, alias_pmt());

      // Reset state
      int nout = d_curr_len;
      d_curr_len = 0;

      return nout;
    }

  } /* namespace echo */
} /* namespace gr */

