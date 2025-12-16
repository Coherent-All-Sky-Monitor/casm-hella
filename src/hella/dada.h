/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <dada_hdu.h>

#ifndef HELLA_DADA_H
#define HELLA_DADA_H

namespace hella {

  /**
   * @brief Connect and lock to a PSRDADA ring buffer as defined in the pipeline info struct.
   *
   * @param p pipeline info struct
   * @return dada_hdu_t* connected PSRDADA HDU
   */
  dada_hdu_t* hdu_connect_read(hella::pinfo_t* p);

  void hdu_cleanup(dada_hdu_t * in);

} // namespace hella

#endif // HELLA_DADA_H
