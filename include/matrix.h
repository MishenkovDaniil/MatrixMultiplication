#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

struct Matrix {
    size_t m;    /* rows */
    size_t n;    /* cols */
    int **arr;   /* arr[row][col] */
};

/* ctors / dtors */
struct Matrix *matrix_ctor(const size_t m, const size_t n);
struct Matrix *matrix_eye(const size_t n);
struct Matrix *matrix_generate(const size_t m, const size_t n, const int max_val);
struct Matrix *matrix_ctor_from_arr(int **arr, const size_t m, const size_t n);
void matrix_dtor(struct Matrix *matrix);

/* ops */
void matrix_fill(struct Matrix *matrix, int val);
void matrix_mul_val(struct Matrix *matrix, int val);

/* single-threaded multiplication */
void mul_matrices_bad2(const struct Matrix *first, const struct Matrix *second, struct Matrix *result);
void mul_matrices_cache_friendly2(const struct Matrix *first, const struct Matrix *second, struct Matrix *result);
void mul_matrices_cache_friendly_most2(const struct Matrix *first, const struct Matrix *second, struct Matrix *result);

/* single-threaded multiplication wrappers */
struct Matrix *mul_matrices_bad(const struct Matrix *A, const struct Matrix *B);
struct Matrix *mul_matrices_cache_friendly(const struct Matrix *A, const struct Matrix *B);
struct Matrix *mul_matrices_cache_friendly_most(const struct Matrix *A, const struct Matrix *B);

/* blocked single-threaded wrapper */
struct Matrix *mul_matrices_blocked(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t block_size);

/* printing */
void matrix_print_row(int *row, const size_t len);
void matrix_print(struct Matrix *matrix, const char *name);

/* multi-threaded multiplication */
/* nthreads == 0 -> auto detect number of processors */
struct Matrix *mul_matrices_pthread(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads);
struct Matrix *mul_matrices_bad_mt(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads);
struct Matrix *mul_matrices_cache_friendly_mt(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads);
struct Matrix *mul_matrices_cache_friendly_most_mt(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads);

/* blocked + pthreads multiplication */
struct Matrix *mul_matrices_blocked_pthread(const struct Matrix *A, const struct Matrix *B, struct Matrix *C, size_t nthreads, size_t block_size);

/* etc */
static inline struct Matrix *eye(size_t n) { return matrix_eye(n); }
static inline void mul_val(struct Matrix *m, int v) { matrix_mul_val(m, v); }
static inline void print_matrix(struct Matrix *m, const char *name) { matrix_print(m, name); }

#endif /* MATRIX_H */
