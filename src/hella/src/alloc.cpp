/***************************************************************************
 *
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Andrew Jameson
 *
 ***************************************************************************/

#include "hella/alloc.h"
#include "hella/macros.h"


void hella::resize_h_scratch(hella::pinfo_t* p, size_t required_size)
{
  if (required_size > p->h_scratch_size)
  {
    checkCuda(cudaFreeHost(p->h_scratch));
    checkCuda(cudaMallocHost(&(p->h_scratch), required_size));
    p->h_scratch_size = required_size;
  }
}

void hella::resize_d_scratch(hella::pinfo_t* p, size_t required_size)
{
  if (required_size > p->d_scratch_size)
  {
    checkCuda(cudaFree(p->d_scratch));
    checkCuda(cudaMalloc(&(p->d_scratch), required_size));
    p->d_scratch_size = required_size;
  }
}

void hella::lock_d_scratch(hella::pinfo_t* p, size_t required_size)
{
  if (p->d_scratch_locked)
  {
    throw std::runtime_error("device scratch space unexpectedly locked");
  }
  resize_d_scratch(p, required_size);
  p->d_scratch_locked = true;
}

void hella::unlock_d_scratch(hella::pinfo_t* p)
{
  if (!p->d_scratch_locked)
  {
    throw std::runtime_error("device scratch space unexpectedly unlocked");
  }
  p->d_scratch_locked = false;
}

void hella::lock_h_scratch(hella::pinfo_t* p, size_t required_size)
{
  if (p->h_scratch_locked)
  {
    throw std::runtime_error("host scratch space unexpectedly locked");
  }
  resize_h_scratch(p, required_size);
  p->h_scratch_locked = true;
}

void hella::unlock_h_scratch(hella::pinfo_t* p)
{
  if (!p->h_scratch_locked)
  {
    throw std::runtime_error("host scratch space unexpectedly unlocked");
  }
  p->h_scratch_locked = false;
}
