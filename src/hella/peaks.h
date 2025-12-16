/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#ifndef HELLA_PEAKS_H
#define HELLA_PEAKS_H

namespace hella
{
  //! to find peaks
  void find_peaks(hella::pinfo_t *p, int bm);

  //! to clear all peaks
  void clear_peaks(hella::pinfo_t *p);

  // output peaks
  void output_peaks(hella::pinfo_t *p, int samp, int restart_socket);

} // namespace hella

#endif // HELLA_PEAKS_H
