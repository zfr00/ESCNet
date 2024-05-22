#ifndef QUERY_CORE_H
#define QUERY_CORE_H

#include <stdint.h>

void _setup_index(
    const int32_t *major_edges,
    const int16_t *major_recon,
    const int16_t *local_recons,
    const int32_t *fid2gmeta,
    int            n_fids,
    const int32_t *major_minor_edges);

void _resolve_path(
    int32_t  fid_src,
    int32_t  fid_dst,
    int      max_hop,
    int32_t *result,
    int64_t *result_size);

void _enlarge_context(int32_t *fids, int n, int m);

#endif