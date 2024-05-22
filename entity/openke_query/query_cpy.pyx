#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: languagelevel=3

from cython.parallel cimport prange

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc
from libc.stdint cimport uintptr_t

ctypedef np.int16_t int16_t
ctypedef np.int32_t int32_t
ctypedef np.int64_t int64_t

cdef extern from "query_core.c":
    pass

cdef extern from "query_core.h":
    void _setup_index(
    const int32_t * major_edges,
    const int16_t * global_recon,
    const int16_t * local_recons,
    const int32_t * fid2gmeta,
    int n_fids,
    const int32_t * major_minor_edges) nogil

    void _resolve_path(
    int32_t  fid_src,
    int32_t  fid_dst,
    int      max_hop,
    int32_t *result,
    int64_t *result_size) nogil

    void _enlarge_context(int32_t *fids, int n, int m) nogil

cpdef enlarge_context(np.ndarray[np.int32_t, ndim=1] fids, int m):
    cdef int n = fids.shape[0]
    if n >= m:
        return fids
    new_fids = np.zeros((m, ), dtype=np.int32)
    new_fids[:n] = fids[:n]
    cdef int32_t[:] new_fids_view = new_fids
    _enlarge_context(&new_fids_view[0], n, m)
    return new_fids

cdef void _query_pairs(
    int n,
    int32_t[:, :] queries,
    int32_t[:, :] results,
    int64_t[:] sizes,
    int max_hop
) nogil:
    cdef int i
    for i in prange(n, schedule='guided'):
        _resolve_path(queries[i, 0], queries[i, 1], max_hop, &results[i, 0], &sizes[i])

cpdef query_pairs(
    np.ndarray[int32_t, ndim=2] queries,
    int max_hop,
):
    cdef int n = queries.shape[0]
    results = np.zeros((n, max_hop), dtype=np.int32)
    cdef int32_t[:, :] results_view = results
    sizes = np.zeros((n), dtype=np.int64)
    cdef int64_t[:] sizes_view = sizes
    cdef int32_t[:, :] queries_view = queries
    _query_pairs(n, queries_view, results_view, sizes_view, max_hop)
    return results, sizes
    

cpdef query_pair(
    int32_t src,
    int32_t dst,
    int max_hop,
):
    result = np.zeros(max_hop, dtype=np.int32)
    cdef int32_t[:] result_view = result
    cdef int64_t result_size
    _resolve_path(src, dst, max_hop, &result_view[0], &result_size)
    return result

_mmap_holder = []

cpdef setup_index(
    np.ndarray[int32_t, ndim=1] major_edges,
    np.ndarray[int16_t, ndim=1] major_recon,
    np.ndarray[int16_t, ndim=1] local_recons,
    np.ndarray[int32_t, ndim=1] fid2gmeta,
    np.ndarray[int32_t, ndim=1] major_minor_edges,
):    
    cdef const int32_t[:] major_edges_view = major_edges
    cdef const int16_t[:] major_recon_view = major_recon
    cdef const int16_t[:] local_recons_view = local_recons
    cdef const int32_t[:] fid2gmeta_view = fid2gmeta
    cdef const int32_t[:] major_minor_edges_view = major_minor_edges

    global _mmap_holder
    _mmap_holder.extend([major_edges, major_recon, local_recons, fid2gmeta, major_minor_edges])

    _setup_index(
        &major_edges_view[0],
        &major_recon_view[0],
        &local_recons_view[0],
        &fid2gmeta_view[0],
        fid2gmeta.shape[0] // 2 - 1,
        &major_minor_edges_view[0],
    )
