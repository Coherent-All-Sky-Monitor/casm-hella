/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <cuda_fp16.h>

#ifndef HELLA_UNUSED_H
#define HELLA_UNUSED_H

namespace hella
{
  //! function to remove time-frequency baseline
  void remove_tf_baseline(hella::pinfo_t* p, half * data, int width, int stride);

  //! function to measure ts using cublas calls
  void blas_ts(half * data, half * unity, half * temp_ts, float * ts, int width, int stride);

  //! input has shape [NBATCH, NCHAN, 15000]
  void apply_batch_test(float * input, float * output, int width, int stride);

} // namespace hella

#endif // HELLA_UNUSED_H
