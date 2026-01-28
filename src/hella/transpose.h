/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <cuda_fp16.h>

#ifndef HELLA_TRANSPOSE_H
#define HELLA_TRANSPOSE_H

namespace hella
{
  // TODO: handle_transpose_input and handle_transpose_output to do transpose via memcpy2d from intermediate array
  /**
   * @brief Unpack and transpose the input 8-bit data to fp16.
   *
   * @param p
   * @param d_data input 8-bit data
   * @param batch output fp16 data
   * @param width
   * @param stride
   */
  void transpose_input_handler(hella::pinfo_t* p, unsigned char * d_data, half * batch, int width, int stride);

  void transpose_output_handler(hella::pinfo_t* p, unsigned char * d_data, half * batch);

} // namespace hella

#endif // HELLA_TRANSPOSE_H
