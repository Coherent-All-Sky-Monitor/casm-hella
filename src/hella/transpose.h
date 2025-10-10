/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include <cuda_fp16.h>

#ifndef HELLA_TRANSPOSE_H
#define HELLA_TRANSPOSE_H

namespace hella
{
  // TODO: handle_transpose_input and handle_transpose_output to do transpose via memcpy2d from intermediate array
  void transpose_input_handler(unsigned char * d_data, half * batch, int width, int stride);

  void transpose_output_handler(unsigned char * d_data, half * batch, int width, int stride);

} // namespace hella

#endif // HELLA_TRANSPOSE_H