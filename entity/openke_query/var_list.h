#ifndef _VAR_LIST_H_
#define _VAR_LIST_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int      nrows;
    int      ncols;
    int *    row_sizes;
    int32_t *data;
} varlist_t;

static inline varlist_t *varlist_new(int nrows, int ncols) {
    varlist_t *ret = (varlist_t *)malloc(sizeof(varlist_t));
    ret->nrows = nrows;
    ret->ncols = ncols;
    ret->row_sizes = (int *)calloc(nrows, sizeof(int));

    ret->data = (int32_t *)malloc(sizeof(int32_t) * nrows * ncols);
    return ret;
}

static inline int32_t varlist_at(varlist_t *v, int i, int j) {
    return v->data[j + i * v->ncols];
}

static inline bool varlist_is_row_full(varlist_t *v, int i) {
    return v->row_sizes[i] == v->ncols;
}

static inline void varlist_push(varlist_t *v, int i, int32_t item) {
    int *size = &v->row_sizes[i];
    v->data[*size + i * v->ncols] = item;
    (*size)++;
}

static inline void varlist_free(varlist_t *v) {
    free(v->row_sizes);
    free(v->data);
    free(v);
}

#endif