#pragma once

#include <cuda_runtime.h>
#include <cassert>

static inline int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

static inline void checkErrors()
{
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

static inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

static inline void checkDeviceSpecs()
{
    int nDevices;
    checkCuda(cudaGetDeviceCount(&nDevices));
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, i));
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Max Blocks Per Multi Processor: %i\n\n", prop.maxBlocksPerMultiProcessor);
        printf("  Multi Processor Count: %i\n\n", prop.multiProcessorCount);
    }
}