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
   * @note uses h and d scratch
   *
   * @param input_data
   * @param width
   * @param stride
   */
  void normalize_data(hella::pinfo_t * p, half * input_data, int width, int stride);

  /**
   * @brief Function to implement bandpass flagging on data
   *
   * @note used h and d scratch
   *
   * @param p pointer to problem info struct
   * @param data data to flag
   * @param width width of the data array in elements
   * @return float
   */
  float bandpass_flag(hella::pinfo_t * p, half * data, int width);

  //! function to bandpass-correct data [uses h and d scratch]
  float bandpass_correct(hella::pinfo_t* p, half * data, int width, int stride);

  //! function to ts-correct data
  void ts_correct(half * data, float * d_ts, int width, int stride);

  /**
   * @brief Apply a single scrunch to the data
   *
   * @note uses h and d scratch
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
  float apply_scrunch(hella::pinfo_t * p, half * data, half * mask, half * d_smooth, float * d_ts, int width, int stride, int tscrunch, int fscrunch, float thresh, int flag, int ts, float * d_flagSpec, int flag1, int flag2);

  // flagger
  // load a batch beam by beam
  // apply all scrunches to the batch
  // unload the batch
  void fast_flagger(hella::pinfo_t * p);

} // namespace hella

#endif // HELLA_FLAGGER_H
