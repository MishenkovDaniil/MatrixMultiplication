#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
static double diff_sec(struct timespec a, struct timespec b) {
    return (a.tv_sec - b.tv_sec) + (a.tv_nsec - b.tv_nsec) / 1e9;
}

int main() {
    int sizes[] = {10, 50, 100, 200, 300, 500, 2000};
    int num_sizes = sizeof(sizes)/sizeof(sizes[0]);

    FILE *fout = fopen("matrix_results.txt", "w");
    if (!fout) {
        perror("fopen");
        return 1;
    }

    srand(time(NULL));

    for (int i = 0; i < num_sizes; ++i)
    {
        printf("[DEBUG] running iteration %d..\n", i);
        int N = sizes[i];
        struct Matrix *A = matrix_generate(N, N, 1000);
        struct Matrix *B = matrix_generate(N, N, 1000);

        double T1, T2, T3, T4, T5;
        struct timespec t0, t1;

        // Algorithm 1
        struct Matrix *C = matrix_ctor(N, N);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mul_matrices_bad2(A, B, C);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        T1 = diff_sec(t1, t0);
        matrix_dtor(C);

        // Algorithm 2
        C = matrix_ctor(N, N);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mul_matrices_cache_friendly2(A, B, C);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        T2 = diff_sec(t1, t0);
        matrix_dtor(C);

        // Algorithm 3
        C = matrix_ctor(N, N);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mul_matrices_cache_friendly_most2(A, B, C);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        T3 = diff_sec(t1, t0);
        matrix_dtor(C);

        // Parallel algorithm
        C = matrix_ctor(N, N);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mul_matrices_cache_friendly_most_mt(A, B, C, 16);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        T4 = diff_sec(t1, t0);
        matrix_dtor(C);

        // Blocked parallel algorithm
        C = matrix_ctor(N, N);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mul_matrices_blocked_pthread(A, B, C, 16, 64);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        T5 = diff_sec(t1, t0);
        matrix_dtor(C);

        fprintf(fout, "{%d, %.6f, %.6f, %.6f, %.6f, %.6f}\n", N, T1, T2, T3, T4, T5);

        matrix_dtor(A);
        matrix_dtor(B);
    }


    fclose(fout);
    printf("Results are stored in matrix_results.txt\n");
    return 0;
}
