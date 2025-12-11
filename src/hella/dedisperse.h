/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#ifndef HELLA_DEDISPERSE_H
#define HELLA_DEDISPERSE_H

#include "hella/definitions.h"

namespace hella
{
  //! function to do all dedispersion stuff
  void dedisperse(pinfo *p, int beam);

  //! return a string describing a dedisp error code
  std::string get_dedisp_error(int dedisp_error_code);

} // namespace hella

#endif // HELLA_DEDISPERSE_H