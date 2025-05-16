#include "HostBuffer.h"

#include <cstdlib>

HostBuffer::HostBuffer(size_t size): data(malloc(size)), size(size) {}
HostBuffer::HostBuffer(void *data, size_t size): data(data), size(size) {}
HostBuffer::~HostBuffer() {
    free(data);
}

