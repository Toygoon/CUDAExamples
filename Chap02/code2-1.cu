#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void helloCuda(void) {
    printf("Hello CUDA from GPU.\n");
}

int main(void) {
    printf("Hello GPU from CPU.\n");
    helloCuda<<<1, 10>>>();

    return 0;
}