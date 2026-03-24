/***************************************************************************
 *
 *   Copyright (C) 2025 Vikram Ravi
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Vikram Ravi, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/dada.h"

#include <ipcbuf.h>
#include <dada_cuda.h>
#include <spdlog/spdlog.h>
#include <sstream>

dada_hdu_t* hella::hdu_connect_read(hella::pinfo_t* p)
{
  // DADA Header plus Data Unit
  dada_hdu_t* hdu{nullptr};

  key_t key = DADA_BLOCK_KEY;
  char* header_in{nullptr};
  uint64_t header_size{0};
  uint64_t block_size{0};

  sscanf(p->inp_path, "%x", &key);

  multilog_t* log = multilog_open("casm_hella", LOG_INFO);
  hdu  = dada_hdu_create(log);
  dada_hdu_set_key(hdu, key);
  dada_hdu_connect(hdu);
  dada_hdu_lock_read(hdu);

  spdlog::trace("Registering data block with CUDA driver");
  dada_cuda_dbregister(hdu);

  // Handle the PSRDADA header
  header_in = ipcbuf_get_next_read(hdu->header_block, &header_size);
  p->dada_header.resize(header_size);
  std::copy_n(header_in, header_size, p->dada_header.begin());
  p->dada_header_parsed = parse_dada_header(p->dada_header);
  ipcbuf_mark_cleared(hdu->header_block);

  block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu->data_block);
  spdlog::debug("Connected to dada buffer header={} bytes block_size={} bytes", header_size, block_size);

  return hdu;
}

std::map<std::string, std::string> hella::parse_dada_header(const std::vector<char> &header)
{
  std::map<std::string, std::string> params{};

  const auto last_header_char = std::find(header.begin(), header.end(), '\0');
  std::string header_str(header.begin(), last_header_char);
  std::stringstream header_ss(header_str);

  std::string line;

  while (std::getline(header_ss, line, '\n'))
  {
    if (line.length() == 0)
    {
      continue;
    }
    auto key_pos = line.find(' ');
    if (key_pos == std::string::npos)
    {
      spdlog::warn("hella::parse_dada_header Got invalid header line ({}), aborting early", line);
      break;
    }
    auto key = line.substr(0, key_pos);
    auto val_pos = line.find_first_not_of(' ', key_pos);
    auto val = line.substr(val_pos);
    params[key] = val;
  }

  return params;
}

void hella::hdu_cleanup(dada_hdu_t * in)
{
  if (dada_hdu_unlock_read (in) < 0)
  {
    spdlog::error("could not unlock read on hdu_in");
  }
  dada_hdu_destroy (in);
}
