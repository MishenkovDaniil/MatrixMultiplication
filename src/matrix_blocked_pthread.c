#define _GNU_SOURCE
#include <pthread.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "matrix.h"

static inline size_t min_sz(size_t a, size_t b) { return a < b ? a : b; }

/* Worker args for blocked parallel multiplication */
struct BlockMtArg {
    const struct Matrix *A;
    const struct Matrix *B;
    struct Matrix *C;
    size_t block_size;
    size_t block_row_begin;
    size_t block_row_end;
};

/* Worker: compute assigned block-rows [block_row_begin, block_row_end) */
static void *block_mt_worker(void *varg)
{
    struct BlockMtArg *arg = (struct BlockMtArg *)varg;
    const struct Matrix *A = arg->A;
    const struct Matrix *B = arg->B;
    struct Matrix *C = arg->C;
    const size_t bs = arg->block_size;

    const size_t M = A->m;
    const size_t K = A->n; // = B->m
    const size_t N = B->n;

    /* iterate over block rows assigned to this thread */
    for (size_t bi = arg->block_row_begin; bi < arg->block_row_end; ++bi) {
        size_t ii = bi * bs;
        size_t i_max = min_sz(ii + bs, M);

        /* for each block column */
        size_t nbj = (N + bs - 1) / bs;
        for (size_t bj = 0; bj < nbj; ++bj) {
            size_t jj = bj * bs;
            size_t j_max = min_sz(jj + bs, N);

            /* run over k-blocks */
            size_t nbk = (K + bs - 1) / bs;
            for (size_t bk = 0; bk < nbk; ++bk) {
                size_t kk = bk * bs;
                size_t k_max = min_sz(kk + bs, K);

                /* inner micro-kernel: for i in ii..i_max, k in kk..k_max, j in jj..j_max:
                   C[i][j] += A[i][k] * B[k][j]
                   We choose order i,k,j for good locality on A and C (and B accessed by k then j).
                */
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        int aik = A->arr[i][k];
                        for (size_t j = jj; j < j_max; ++j) {
                            C->arr[i][j] += aik * B->arr[k][j];
                        }
                    }
                }
            }
        }
    }

    return NULL;
}

/* Generic blocked multi-threaded implementation */
struct Matrix *mul_matrices_blocked_pthread(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads, size_t block_size)
{
    assert(A && B);
    assert(A->n == B->m);

    if (block_size == 0) {
        block_size = 32;
    }

    if (nthreads == 0) {
        long procs = sysconf(_SC_NPROCESSORS_ONLN);
        if (procs > 0) nthreads = (size_t)procs;
        else nthreads = 1;
    }
    if (nthreads == 0) nthreads = 1;

    /* quick single-thread fallback (avoid thread overhead for small work) */
    if (nthreads == 1) {
        /* emulate worker over all block rows */
        struct BlockMtArg single = { A, B, C, block_size, 0, (A->m + block_size - 1) / block_size };
        block_mt_worker(&single);
        return C;
    }

    size_t block_rows = (A->m + block_size - 1) / block_size;
    if (block_rows == 0) block_rows = 1;

    pthread_t *threads = (pthread_t *)calloc(nthreads, sizeof(pthread_t));
    assert(threads);
    struct BlockMtArg *args = (struct BlockMtArg *)calloc(nthreads, sizeof(struct BlockMtArg));
    assert(args);

    /* distribute block_rows among threads */
    size_t base = block_rows / nthreads;
    size_t rem = block_rows % nthreads;
    size_t cur_block = 0;
    for (size_t t = 0; t < nthreads; ++t) {
        size_t my_blocks = base + (t < rem ? 1 : 0);
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].block_size = block_size;
        args[t].block_row_begin = cur_block;
        args[t].block_row_end = cur_block + my_blocks;
        cur_block += my_blocks;

        if (args[t].block_row_begin < args[t].block_row_end) {
            int rc = pthread_create(&threads[t], NULL, block_mt_worker, &args[t]);
            if (rc != 0) {
                /* fallback run in main thread */
                block_mt_worker(&args[t]);
                threads[t] = 0;
            }
        } else {
            threads[t] = 0;
        }
    }

    for (size_t t = 0; t < nthreads; ++t) {
        if (threads[t]) pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
    return C;
}

/* Single-threaded blocked wrapper */
struct Matrix *mul_matrices_blocked(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t block_size)
{
    return mul_matrices_blocked_pthread(A, B, C, 1, block_size);
}
