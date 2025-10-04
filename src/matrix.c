#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "matrix.h"

struct Matrix *matrix_ctor(const size_t m, const size_t n)
{
    assert(m && n);
    struct Matrix *matrix = (struct Matrix *)calloc(1, sizeof(struct Matrix));
    assert(matrix);

    matrix->m = m;
    matrix->n = n;
    matrix->arr = (int **)calloc(m, sizeof(int *));
    assert(matrix->arr);

    for (size_t row_id = 0; row_id < m; ++row_id)
    {
        matrix->arr[row_id] = (int *)calloc(n, sizeof(int));
        assert(matrix->arr[row_id]);
    }

    return matrix;
}

struct Matrix *matrix_eye(const size_t n)
{
    assert(n);

    struct Matrix *matrix = (struct Matrix *)calloc(1, sizeof(struct Matrix));
    assert(matrix);

    matrix->m = n;
    matrix->n = n;
    matrix->arr = (int **)calloc(n, sizeof(int *));
    assert(matrix->arr);

    for (size_t row_id = 0; row_id < n; ++row_id)
    {
        matrix->arr[row_id] = (int *)calloc(n, sizeof(int));
        assert(matrix->arr[row_id]);

        matrix->arr[row_id][row_id] = 1;
    }

    return matrix;
}

struct Matrix *matrix_generate(const size_t m, const size_t n, const int max_val)
{
    assert(n && m);
    struct Matrix *matrix = (struct Matrix *)calloc(1, sizeof(struct Matrix));
    assert(matrix);

    matrix->m = m;
    matrix->n = n;
    matrix->arr = (int **)calloc(m, sizeof(int *));
    assert(matrix->arr);

    for (size_t row_id = 0; row_id < m; ++row_id)
    {
        matrix->arr[row_id] = (int *)calloc(n, sizeof(int));
        assert(matrix->arr[row_id]);
        for (size_t id = 0; id < n; ++id)
        {
            matrix->arr[row_id][id] = rand() % max_val;
        }
    }

    return matrix;
}

struct Matrix *matrix_ctor_from_arr(int **arr, const size_t m, const size_t n)
{
    assert(arr && n && m);

    struct Matrix *matrix = (struct Matrix *)calloc(1, sizeof(struct Matrix));
    assert(matrix);

    matrix->m = n;
    matrix->n = n;
    matrix->arr = arr;

    return matrix;
}

void matrix_dtor(struct Matrix *matrix)
{
    assert(matrix && matrix->arr);

    for (size_t row_id = 0; row_id < matrix->m; ++row_id)
    {
        free(matrix->arr[row_id]);
    }
    free(matrix->arr);
    free(matrix);
}

void matrix_fill(struct Matrix *matrix, int val)
{
    assert(matrix);
    for (size_t i = 0; i < matrix->m; ++i)
    {
        for (size_t j = 0; j < matrix->n; ++j)
        {
            matrix->arr[i][j] = val;
        }
    }
}

void matrix_mul_val(struct Matrix *matrix, int val)
{
    assert(matrix);

    for (size_t i = 0; i < matrix->m; ++i)
    {
        for (size_t j = 0; j < matrix->n; ++j)
        {
            matrix->arr[i][j] *= val;
        }
    }
}

void mul_matrices_bad2(const struct Matrix *first, const struct Matrix *second, struct Matrix *result)
{
    assert(first && second && result);
    assert(first->n == second->m && first->m == result->m && second->n == result->n);

    const size_t intermediate = first->n; // or second->m
    /* Worst cache-friendliness for memory row-based arrays. */
    for (size_t j = 0; j < result->n; ++j)
    {
        for (size_t k = 0; k < intermediate; ++k)
        {
            for (size_t i = 0; i < result->m; ++i)
            {
                result->arr[i][j] += first->arr[i][k] * second->arr[k][j];
            }
        }
    }
}

void mul_matrices_cache_friendly2(const struct Matrix *first, const struct Matrix *second, struct Matrix *result)
{
    assert(first && second && result);
    assert(first->n == second->m && first->m == result->m && second->n == result->n);

    const size_t intermediate = first->n; // or second->m
    /* Here second matrix is index non-cache-friendly, but first and C are. */
    /* Good when first >> second. */
    for (size_t i = 0; i < result->m; ++i)
    {
        for (size_t j = 0; j < result->n; ++j)
        {
            for (size_t k = 0; k < intermediate; ++k)
            {
                result->arr[i][j] += first->arr[i][k] * second->arr[k][j];
            }
        }
    }
}

void mul_matrices_cache_friendly_most2(const struct Matrix *first, const struct Matrix *second, struct Matrix *result)
{
    assert(first && second && result);
    assert(first->n == second->m && first->m == result->m && second->n == result->n);

    const size_t intermediate = first->n; // or second->m
    /* Here first matrix is index non-cache-friendly, but second and C are. */
    /* Good when first << second. */
    for (size_t i = 0; i < result->m; ++i)
    {
        for (size_t k = 0; k < intermediate; ++k)
        {
            for (size_t j = 0; j < result->n; ++j)
            {
                result->arr[i][j] += first->arr[i][k] * second->arr[k][j];
            }
        }
    }
}

struct Matrix *mul_matrices_bad(const struct Matrix *A, const struct Matrix *B)
{
    assert(A && B);
    assert(A->n == B->m);

    struct Matrix *C = matrix_ctor(A->m, B->n);
    assert(C);

    mul_matrices_bad2(A, B, C);
    return C;
}

struct Matrix *mul_matrices_cache_friendly(const struct Matrix *A, const struct Matrix *B)
{
    assert(A && B);
    assert(A->n == B->m);

    struct Matrix *C = matrix_ctor(A->m, B->n);
    assert(C);

    mul_matrices_cache_friendly2(A, B, C);
    return C;
}

struct Matrix *mul_matrices_cache_friendly_most(const struct Matrix *A, const struct Matrix *B)
{
    assert(A && B);
    assert(A->n == B->m);

    struct Matrix *C = matrix_ctor(A->m, B->n);
    assert(C);

    mul_matrices_cache_friendly_most2(A, B, C);
    return C;
}

void matrix_print_row(int *row, const size_t len)
{
    assert(row);
    for(size_t i = 0; i < len; ++i)
    {
        printf("%d\t", row[i]);
    }
    printf("\n");
}

void matrix_print(struct Matrix *matrix, const char *name)
{
    assert(matrix && name);
    printf("[Matrix %s]\n", name);
    for(size_t row_id = 0; row_id < matrix->m; ++row_id)
    {
        matrix_print_row(matrix->arr[row_id], matrix->n);
    }
}