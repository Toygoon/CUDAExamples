#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void checkDeviceMemory(void) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory (free/total) = %lld%lld bytes\n", free, total);
}

int main(void) {
    int* dDataPtr;
    cudaError_t errCode;

    checkDeviceMemory();
    errCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
    printf("cudaMalloc - %s\n", cudaGetErrorName(errCode));
    checkDeviceMemory();

    errCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
    printf("cudaMemset - %s\n", cudaGetErrorName(errCode);

    errCode = cudaFree(dDataPtr);
    printf("cudaFree - %s\n", cudaGetErrorName(errCode));
    checkDeviceMemory();

    return 0;
}