/***************************************************************************
 *
 *   Copyright (C) 2025 TBA
 *   Copyright (C) 2025 Fourier Space
 *   Authors: TBA, Andrew Jameson
 *
 ***************************************************************************/

#include "hella/dada.h"

#include "ipcbuf.h"
#include <spdlog/spdlog.h>

dada_hdu_t* hella::hdu_connect_read(pinfo_t* p)
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

  // page all of the input data block buffers into RAM
  ipcbuf_page(reinterpret_cast<ipcbuf_t*>(hdu->data_block));

  header_in = ipcbuf_get_next_read(hdu->header_block, &header_size);
  ipcbuf_mark_cleared(hdu->header_block);
  block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu->data_block);
  spdlog::debug("Connected to dada buffer header={} bytes block_size={} bytes", header_size, block_size);

  return hdu;
}

void hella::hdu_cleanup(dada_hdu_t * in)
{
  if (dada_hdu_unlock_read (in) < 0)
  {
    spdlog::error("could not unlock read on hdu_in");
  }
  dada_hdu_destroy (in);
}
