/***************************************************************************
 *
 *   Copyright (C) 2025 Fourier Space
 *   Authors: Andrew Jameson
 *
 ***************************************************************************/

#include "hella/definitions.h"
#include "hella/macros.h"

#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>

namespace hella
{
  template <typename T>
  void alloc_cpu(T ** allocation, size_t num_elements)
  {
    size_t required_bytes = sizeof(T) * num_elements;
    spdlog::trace("alloc_cpu requested num_elements={} element_size={} -> {} bytes", num_elements, sizeof(T), required_bytes);
    *allocation = reinterpret_cast<T*>(malloc(required_bytes));
    if (!*allocation)
    {
      std::ostringstream oss;
      oss << "failed to allocated " << required_bytes << " of cpu memory";
      throw std::runtime_error(oss.str());
    }
  }

  template <typename T>
  void release_cpu(T ** allocation)
  {
    if (*allocation != nullptr)
    {
      free(*allocation);
    }
    *allocation = nullptr;
  }

  template <typename T>
  void alloc_cpu_host(T ** allocation, size_t num_elements)
  {
    size_t required_bytes = sizeof(T) * num_elements;
    spdlog::trace("alloc_cpu_host requested num_elements={} element_size={} -> {} bytes", num_elements, sizeof(T), required_bytes);
    checkCuda(cudaMallocHost(allocation, required_bytes));
  }

  template <typename T>
  void release_cpu_host(T ** allocation)
  {
    checkCuda(cudaFreeHost(*allocation));
    *allocation = nullptr;
  }

  template <typename T>
  void alloc_gpu(T ** device_ptr, size_t num_elements)
  {
    size_t required_bytes = sizeof(T) * num_elements;
    checkCuda(cudaMalloc(device_ptr, required_bytes));
  }

  template <typename T>
  void release_gpu(T ** allocation)
  {
    checkCuda(cudaFree(*allocation));
    *allocation = nullptr;
  }

  template <typename T>
  void alloc_gpu_pitch(T ** device_ptr, size_t * pitch, size_t width, size_t height)
  {
    checkCuda(cudaMallocPitch(reinterpret_cast<void**>(device_ptr), pitch, sizeof(T) * width, height));
  }

  void lock_h_scratch(pinfo* p, size_t required_size);
  void resize_h_scratch(pinfo* p, size_t required_size);
  void unlock_h_scratch(pinfo* p);

  void lock_d_scratch(pinfo* p, size_t required_size);
  void resize_d_scratch(pinfo* p, size_t required_size);
  void unlock_d_scratch(pinfo* p);

} // namespace hella