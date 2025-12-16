
/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <cuda_fp16.h>

#ifndef HELLA_TIMESERIES_H
#define HELLA_TIMESERIES_H

namespace hella
{
  void ts_correct(half * data, float * d_ts, int width, int stride);

  //! function to orchestrate host median filtering of time series
  void med_filter_ts(hella::pinfo_t* p, float * d_ts, int width);

} // namespace hella

#endif // HELLA_TIMESERIES_H
