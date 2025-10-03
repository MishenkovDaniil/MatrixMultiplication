#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include "matrix.h"

#define N 1000

static void run_example1(int n)
{
    struct Matrix *A = matrix_generate(n, n, 1000);
    struct Matrix *B = matrix_generate(n, n, 1000);

    matrix_print(A, "first");
    matrix_print(B, "second");

    struct Matrix *C = mul_matrices_cache_friendly_most(A, B);

    matrix_print(C, "result");

    matrix_dtor(C);
    matrix_dtor(B);
    matrix_dtor(A);
}

static void run_example2(int n)
{
    struct Matrix *A = matrix_generate(N, N, 1000);
    struct Matrix *B = matrix_generate(N, N, 1000);
    struct Matrix *C = matrix_ctor(N, N);

    clock_t start = clock();

    mul_matrices_bad2(A, B, C);

    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;

    printf("[Bad] N = %d, Elapsed time = %f.\n", N, seconds);
    matrix_fill(C, 0);
    start = clock();

    mul_matrices_cache_friendly2(A, B, C);

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;

    printf("[cache-friendly] N = %d, Elapsed time = %f.\n", N, seconds);

    matrix_fill(C, 0);
    start = clock();

    mul_matrices_cache_friendly_most2(A, B, C);

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;

    printf("[most cache-friendly] N = %d, Elapsed time = %f.\n", N, seconds);

    matrix_dtor(C);
    matrix_dtor(B);
    matrix_dtor(A);
}

int main()
{
    printf("Run simple multiplication of matrices..\n");
    run_example1(5);

    printf("Run different multiplication strategies..\n");
    run_example2(N);
    return 0;
}