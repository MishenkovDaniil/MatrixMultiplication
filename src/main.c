#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include "matrix.h"

#define N 1000

static double diff_sec(struct timespec a, struct timespec b) {
    return (a.tv_sec - b.tv_sec) + (a.tv_nsec - b.tv_nsec) / 1e9;
}

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

    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    mul_matrices_bad2(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double seconds = diff_sec(t1, t0);

    printf("[Bad] N = %d, Elapsed time = %lf.\n", N, seconds);
    matrix_fill(C, 0);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    mul_matrices_cache_friendly2(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    seconds = diff_sec(t1, t0);

    printf("[cache-friendly] N = %d, Elapsed time = %lf.\n", N, seconds);

    matrix_fill(C, 0);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    mul_matrices_cache_friendly_most2(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    seconds = diff_sec(t1, t0);

    printf("[most cache-friendly] N = %d, Elapsed time = %lf.\n", N, seconds);

    matrix_fill(C, 0);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    mul_matrices_cache_friendly_most_mt(A, B, C,8);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    seconds = diff_sec(t1, t0);

    printf("[Parallel] N = %d, Elapsed time = %lf.\n", N, seconds);

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