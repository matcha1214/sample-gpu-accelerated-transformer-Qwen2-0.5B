#pragma once

#include <cstddef>

// Fixed-size buffer stored in host (CPU) memory, automatically free'd on destruct
class HostBuffer {
public:
    /// Allocate new memory
    explicit HostBuffer(size_t size);
    /// data: previously allocated with malloc
    HostBuffer(void* data, size_t size);
    ~HostBuffer();

    void *data{};
    size_t size{};
};