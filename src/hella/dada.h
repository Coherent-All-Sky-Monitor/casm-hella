/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"

#include <dada_hdu.h>

#include <map>
#include <vector>
#include <string>

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

  /**
   * @brief Parse a raw PSRDADA header into a map of key-value pairs
   *
   * @param header the raw header
   * 
   * @return the parsed key-value pairs
   */
  std::map<std::string, std::string> parse_dada_header(const std::vector<char> &header);

} // namespace hella

#endif // HELLA_DADA_H
