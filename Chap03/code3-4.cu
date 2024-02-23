#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

int main(void) {
    int mem_size = sizeof(int) * NUM_DATA;
    int *a = new int[NUM_DATA], *b = new int[NUM_DATA], *c = new int[NUM_DATA];

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c[i] = a[i] + b[i];
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}