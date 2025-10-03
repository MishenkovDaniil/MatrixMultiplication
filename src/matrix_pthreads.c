#define _GNU_SOURCE
#include <pthread.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "matrix.h"

/* ---------------- worker arg ---------------- */
struct MtArg {
    const struct Matrix *A;
    const struct Matrix *B;
    struct Matrix *C;
    size_t row_begin;   // inclusive
    size_t row_end;     // exclusive
    int order;          // 0 = bad (j,k,i), 1 = cache_friendly (i,j,k), 2 = cache_friendly_most (i,k,j)
};

/* Worker: computes rows [row_begin, row_end) of C */
static void *mt_worker(void *varg)
{
    struct MtArg *arg = (struct MtArg *)varg;
    const struct Matrix *A = arg->A;
    const struct Matrix *B = arg->B;
    struct Matrix *C = arg->C;
    const size_t n = C->n;
    const size_t kdim = A->n;

    if (arg->order == 0) {
        /* bad ordering: j,k,i  (as in mul_matrices_bad2) */
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < kdim; ++k) {
                for (size_t i = arg->row_begin; i < arg->row_end; ++i) {
                    C->arr[i][j] += A->arr[i][k] * B->arr[k][j];
                }
            }
        }
    } else if (arg->order == 1) {
        /* cache friendly: i,j,k */
        for (size_t i = arg->row_begin; i < arg->row_end; ++i) {
            for (size_t j = 0; j < n; ++j) {
                int sum = 0;
                for (size_t k = 0; k < kdim; ++k) {
                    sum += A->arr[i][k] * B->arr[k][j];
                }
                C->arr[i][j] = sum;
            }
        }
    } else {
        /* most cache friendly: i,k,j */
        for (size_t i = arg->row_begin; i < arg->row_end; ++i) {
            for (size_t k = 0; k < kdim; ++k) {
                int aik = A->arr[i][k];
                for (size_t j = 0; j < n; ++j) {
                    C->arr[i][j] += aik * B->arr[k][j];
                }
            }
        }
    }

    return NULL;
}

static struct Matrix *mul_matrices_pthread_generic(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads, int order)
{
    assert(A && B);
    assert(A->n == B->m);

    if (nthreads == 0) {
        long procs = sysconf(_SC_NPROCESSORS_ONLN);
        if (procs > 0) nthreads = (size_t)procs;
        else nthreads = 1;
    }
    if (nthreads == 0) nthreads = 1;

    // If nthreads is 1, fallback to single-threaded kernel (to avoid thread overhead)
    if (nthreads == 1) {
        struct MtArg arg = { A, B, C, 0, C->m, order };
        mt_worker(&arg);
        return C;
    }

    pthread_t *threads = (pthread_t *)calloc(nthreads, sizeof(pthread_t));
    assert(threads);
    struct MtArg *args = (struct MtArg *)calloc(nthreads, sizeof(struct MtArg));
    assert(args);

    // Partition rows as evenly as possible
    size_t rows = C->m;
    size_t base = rows / nthreads;
    size_t rem = rows % nthreads;
    size_t cur = 0;

    for (size_t t = 0; t < nthreads; ++t) {
        size_t chunk = base + (t < rem ? 1 : 0);
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_begin = cur;
        args[t].row_end = cur + chunk;
        args[t].order = order;
        cur += chunk;
        // create thread only if it has non-empty range
        if (args[t].row_begin < args[t].row_end) {
            int rc = pthread_create(&threads[t], NULL, mt_worker, &args[t]);
            if (rc != 0) {
                // fallback: run in main thread if create failed
                mt_worker(&args[t]);
                threads[t] = 0; // mark as not created
            }
        } else {
            threads[t] = 0;
        }
    }

    // join threads
    for (size_t t = 0; t < nthreads; ++t) {
        if (threads[t]) {
            pthread_join(threads[t], NULL);
        }
    }

    free(threads);
    free(args);
    return C;
}

/* Public wrappers */

/* Generic: default order most cache-friendly (i,k,j) */
struct Matrix *mul_matrices_pthread(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads)
{
    return mul_matrices_pthread_generic(A, B, C, nthreads, 2);
}

/* Bad ordering parallel (j,k,i) */
struct Matrix *mul_matrices_bad_mt(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads)
{
    return mul_matrices_pthread_generic(A, B, C, nthreads, 0);
}

/* i,j,k ordering parallel */
struct Matrix *mul_matrices_cache_friendly_mt(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads)
{
    return mul_matrices_pthread_generic(A, B, C, nthreads, 1);
}

/* i,k,j ordering parallel (most cache-friendly) */
struct Matrix *mul_matrices_cache_friendly_most_mt(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads)
{
    return mul_matrices_pthread_generic(A, B, C, nthreads, 2);
}
