/**
 * Copyright Andrew Jameson
 *
 */

#include <cstring>

namespace hella
{
  template typename<T>
  T* alloc_cpu<T>(size_t num_elements)
  {
    size_t required_bytes = sizeof(T) * num_elements;
    T* allocated = malloc(required_bytes);
    if (!allocated)
    {
      throw std::bad_alloc("failed to allocated " + std::tostring(required_bytes) + " of cpu memory");
    }
    return allocated;
  }

  template typename<T>
  void release_cpu<T>(T ** allocation)
  {
    if (*allocation != nullptr)
    {
      free(*allocation);
    }
    *allocation = nullptr;
  }

} // namespace hella