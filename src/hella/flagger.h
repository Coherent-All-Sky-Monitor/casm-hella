/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <cuda_fp16.h>

#ifndef HELLA_FLAGGER_H
#define HELLA_FLAGGER_H

namespace hella
{
  /**
   * @brief function to normalize data, assume initial mean is 1
   *
   * @param data
   * @param width
   * @param stride
   */
  void normalize_data(half * data, int width, int stride);

  float calculate_stddev(half * d_data, int width, int height, int stride);

  //! function to implement bandpass flagging on data
  float bandpass_flag(pinfo * p, half * data);

  //! function to bandpass-correct data
  float bandpass_correct(half * data, int width, int stride);

  //! function to ts-correct data
  void ts_correct(half * data, float * d_ts, int width, int stride);

  /**
   * @brief Apply a single scrunch to the data
   *
   * @param p
   * @param data
   * @param mask
   * @param d_smooth
   * @param d_ts
   * @param width
   * @param stride
   * @param tscrunch
   * @param fscrunch
   * @param thresh
   * @param flag
   * @param ts
   * @param d_flagSpec
   * @param flag1
   * @param flag2
   * @return float
   */
  float apply_scrunch(pinfo * p, half * data, half * mask, half * d_smooth, float * d_ts, int width, int stride, int tscrunch, int fscrunch, float thresh, int flag, int ts, float * d_flagSpec, int flag1, int flag2);

  // flagger
  // load a batch beam by beam
  // apply all scrunches to the batch
  // unload the batch
  void fast_flagger(pinfo * p);

} // namespace hella

#endif // HELLA_FLAGGER_H