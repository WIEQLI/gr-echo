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


#ifndef INCLUDED_ECHO_PDU_COMPLEX_TO_ETTUS_TAGGED_STREAM_H
#define INCLUDED_ECHO_PDU_COMPLEX_TO_ETTUS_TAGGED_STREAM_H

#include <echo/api.h>
#include <gnuradio/blocks/pdu.h>
#include <gnuradio/tagged_stream_block.h>

namespace gr {
  namespace echo {

    /*!
     * \brief <+description of block+>
     * \ingroup echo
     *
     */
    class ECHO_API pdu_complex_to_ettus_tagged_stream : virtual public gr::tagged_stream_block
    {
     public:
      typedef boost::shared_ptr<pdu_complex_to_ettus_tagged_stream> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of echo::pdu_complex_to_ettus_tagged_stream.
       *
       * To avoid accidental use of raw pointers, echo::pdu_complex_to_ettus_tagged_stream's
       * constructor is in a private implementation
       * class. echo::pdu_complex_to_ettus_tagged_stream::make is the public interface for
       * creating new instances.
       */
      static sptr make(const std::string& length_tag_name, const std::string& sob_tag_name, const std::string& eob_tag_name, int timeout_ms);
    };

  } // namespace echo
} // namespace gr

#endif /* INCLUDED_ECHO_PDU_COMPLEX_TO_ETTUS_TAGGED_STREAM_H */

