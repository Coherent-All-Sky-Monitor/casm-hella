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
  void find_peaks(pinfo *p, int bm);

  //! to clear all peaks
  void clear_peaks(pinfo *p);

  // output peaks
  void output_peaks(pinfo *p, int samp, int restart_socket);

} // namespace hella

#endif // HELLA_PEAKS_H