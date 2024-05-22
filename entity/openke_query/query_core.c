#include "query_core.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static int n_groups;
static int n_major_groups;
static int n_fids;

static const int32_t * m_major_edges;
static const int16_t * m_major_recon;
static int *           m_local_sizes;
static const int16_t **m_local_recons;
static const int32_t * m_fid2gmeta;
static int *           m_major_minor_edge_sizes;
static const int32_t **m_major_minor_edges;
static int32_t **      m_gmeta2fid;

#define FID2GID(fid) (m_fid2gmeta[2 * (fid)])
#define FID2GMETA(fid) (m_fid2gmeta[2 * (fid) + 1])
#define GMETA2FID(gid, lid) (m_gmeta2fid[gid][lid])
#define IS_GID_MAJOR(gid) ((gid) < n_major_groups)
#define LOCAL_SIZE(gid) (m_local_sizes[gid])

#define LOCAL_RECON(gid, u, v)                                                 \
    (m_local_recons[gid][(v) + m_local_sizes[gid] * (u)])

#define STORE_FID_AND_CHECK()                                                  \
    do {                                                                       \
        result[(*result_size)] = fid;                                          \
        if (++(*result_size) == max_hop) {                                     \
            return;                                                            \
        }                                                                      \
    } while (0)

#define MAJOR_RECON(gid1, gid2) (m_major_recon[(gid1)*n_major_groups + (gid2)])

#define MAJOR_EDGE(gid1, gid2, offset)                                         \
    (m_major_edges[offset + 2 * (gid2) + 2 * n_major_groups * (gid1)])

void _setup_index(
    const int32_t *major_edges,
    const int16_t *major_recon,
    const int16_t *local_recons,
    const int32_t *fid2gmeta,
    int            nfids,
    const int32_t *major_minor_edges) {
    n_groups = fid2gmeta[0];
    n_major_groups = fid2gmeta[1];
    n_fids = nfids;

    printf(
        "n_groups=%d n_major_groups=%d n_fids=%d\n",
        n_groups,
        n_major_groups,
        n_fids);

    m_major_edges = major_edges;
    m_major_recon = major_recon;

    m_local_sizes = malloc(sizeof(int) * n_groups);
    m_local_recons = malloc(sizeof(int16_t *) * n_groups);
    int i, size;
    for (i = 0; i < n_groups; i++) {
        m_local_sizes[i] = size = local_recons[0];
        local_recons++;
        m_local_recons[i] = local_recons;
        local_recons += size * size;
    }
    m_fid2gmeta = fid2gmeta + 2;

    int n_minor_groups = n_groups - n_major_groups;
    m_major_minor_edge_sizes = malloc(sizeof(int) * n_minor_groups);
    m_major_minor_edges = malloc(sizeof(int32_t *) * n_minor_groups);
    for (i = 0; i < n_minor_groups; i++) {
        m_major_minor_edge_sizes[i] = size = major_minor_edges[0];
        major_minor_edges++;
        m_major_minor_edges[i] = major_minor_edges;
        major_minor_edges += 3 * size;
    }

    m_gmeta2fid = malloc(n_groups * sizeof(int32_t *));
    for (i = 0; i < n_groups; i++) {
        m_gmeta2fid[i] = malloc(m_local_sizes[i] * sizeof(int32_t));
    }
    int fid;
    for (fid = 1; fid < n_fids; fid++) {
        int32_t gid = FID2GID(fid);
        if (gid >= 0) {
            m_gmeta2fid[gid][FID2GMETA(fid)] = fid;
        }
    }
}

typedef struct {
    int32_t gid;
    int16_t lid;
} group_coord_t;

static inline bool find_mm_edge(
    int32_t        gid,
    group_coord_t *major_coord,
    group_coord_t *minor_coord) {
    int32_t minor_gid = gid - n_major_groups;
    assert(minor_gid >= 0);
    int size = m_major_minor_edge_sizes[minor_gid];
    if (size == 0) {
        return false;
    }
    int idx = rand() % size;
    major_coord->gid = m_major_minor_edges[minor_gid][idx * 3];
    major_coord->lid = m_major_minor_edges[minor_gid][idx * 3 + 1];
    minor_coord->gid = gid;
    minor_coord->lid = m_major_minor_edges[minor_gid][idx * 3 + 2];
    return true;
}

static inline void _resolve_in_group(
    group_coord_t in,
    group_coord_t out,
    int           max_hop,
    int32_t *     result,
    int64_t *     result_size) {
    assert(in.gid == out.gid);
    int32_t gid = in.gid;
    int32_t lid = in.lid;
    int32_t lid_dst = out.lid;
    int32_t fid;
    while (lid != lid_dst) {
        fid = GMETA2FID(gid, lid);
        STORE_FID_AND_CHECK();
        lid = LOCAL_RECON(gid, lid, lid_dst);
    }
    fid = GMETA2FID(gid, lid);
    STORE_FID_AND_CHECK();
}
#define IS_FULL() (*result_size == max_hop)
#define RESOLVE_IN_GROUP(in, out)                                              \
    do {                                                                       \
        _resolve_in_group(in, out, max_hop, result, result_size);              \
        if (IS_FULL())                                                         \
            return;                                                            \
    } while (0)

void _resolve_path(
    int32_t  fid_src,
    int32_t  fid_dst,
    int      max_hop,
    int32_t *result,
    int64_t *result_size) {

    int32_t fid = fid_src;
    int32_t gid = FID2GID(fid);
    if (gid == -2) { // entity with no edge
        goto EXIT;
    }
    if (gid == -1) {
        STORE_FID_AND_CHECK();
        fid = FID2GMETA(fid);

        gid = FID2GID(fid);
        assert(gid >= 0);
    }
    // now src in a group

    int32_t gid_dst = FID2GID(fid_dst);
    int32_t fid_dst_orig = fid_dst;
    bool    dst_is_single = false;

    if (gid_dst == -2) {
        goto EXIT;
    }
    if (gid_dst == -1) {
        dst_is_single = true;
        fid_dst = FID2GMETA(fid_dst);
        gid_dst = FID2GID(fid_dst);
        assert(gid_dst >= 0);
    }
    // now dst in a group

    group_coord_t minor_out_dst = {.gid = -1};
    group_coord_t minor_in_dst = {.gid = -1};
    group_coord_t coord_dst = {.gid = gid_dst, .lid = FID2GMETA(fid_dst)};
    group_coord_t coord_src = {.gid = gid, .lid = FID2GMETA(fid)};

    if (IS_GID_MAJOR(gid) && IS_GID_MAJOR(gid_dst)) {
        // pass
    } else if (IS_GID_MAJOR(gid) /*&&!IS_GID_MAJOR(gid_dst)*/) {
        minor_out_dst = coord_dst;
        if (!find_mm_edge(gid_dst, &coord_dst, &minor_in_dst)) {
            goto EXIT;
        }
    } else if (/*!IS_GID_MAJOR(gid)&&*/ IS_GID_MAJOR(gid_dst)) {
        group_coord_t minor_in_src = coord_src;
        group_coord_t minor_out_src;
        if (!find_mm_edge(gid, &coord_src, &minor_out_src)) {
            goto EXIT;
        }
        RESOLVE_IN_GROUP(minor_in_src, minor_out_src);
    } else {
        if (coord_src.gid == coord_dst.gid) {
            minor_in_dst = coord_src;
            minor_out_dst = coord_dst;
            goto FINALIZE;
        } else {
            group_coord_t minor_in_src = coord_src;
            group_coord_t minor_out_src;
            if (!find_mm_edge(gid, &coord_src, &minor_out_src)) {
                goto EXIT;
            }
            RESOLVE_IN_GROUP(minor_in_src, minor_out_src);

            minor_out_dst = coord_dst;
            if (!find_mm_edge(gid_dst, &coord_dst, &minor_in_dst)) {
                goto EXIT;
            }
        }
    }

    if (MAJOR_RECON(coord_src.gid, coord_dst.gid) == -1) {
        goto EXIT;
    }

    // now we're in a group and surely connected with dst

    for (;;) {
        group_coord_t coord_src_next;
        group_coord_t coord_src_out;
        bool          break_after_this_iter = false;
        if (coord_src.gid == coord_dst.gid) {
            break_after_this_iter = true;
            coord_src_out = coord_dst;
        } else {
            int32_t next_gid = MAJOR_RECON(coord_src.gid, coord_dst.gid);
            coord_src_next.gid = next_gid;
            coord_src_out.gid = coord_src.gid;
            coord_src_out.lid = MAJOR_EDGE(coord_src.gid, next_gid, 0);
            coord_src_next.lid = MAJOR_EDGE(coord_src.gid, next_gid, 1);
        }
        RESOLVE_IN_GROUP(coord_src, coord_src_out);
        if (break_after_this_iter) {
            break;
        }
        coord_src = coord_src_next;
    }

FINALIZE:
    if (minor_in_dst.gid != -1) {
        RESOLVE_IN_GROUP(minor_in_dst, minor_out_dst);
    }

    if (dst_is_single) {
        fid = fid_dst_orig;
        STORE_FID_AND_CHECK();
    }

    return;

EXIT:
    if (*result_size == 0)
        STORE_FID_AND_CHECK();
    return;
}

#include "var_list.h"

typedef unsigned char hash512_t[64];

static inline void hash512_setbit(hash512_t h, int n) {
    int q = n / 8;
    int r = n % 8;
    (h)[q] |= (1 << r);
}

static inline bool hash512_hasbit(hash512_t h, int n) {
    int q = n / 8;
    int r = n % 8;
    return ((h)[q] & (1 << r)) != 0;
}

#include <string.h>

static inline void hash512_clear(hash512_t h) { memset(h, 0, 64); }

static inline int min(int x, int y) { return x < y ? x : y; }

void _enlarge_context(int32_t *fids, int n, int m) {
#define RETURN_IF_FULL()                                                       \
    do {                                                                       \
        if (n == m)                                                            \
            goto EXIT;                                                         \
    } while (0)
#define PUSH_CONTEXT(X)                                                        \
    do {                                                                       \
        int32_t tmp_fid = (X);                                                 \
        int     tmp;                                                           \
        bool    exists = false;                                                \
        for (tmp = 0; tmp < n; tmp++) {                                        \
            if (fids[tmp] == tmp_fid) {                                        \
                exists = true;                                                 \
                break;                                                         \
            }                                                                  \
        }                                                                      \
        if (!exists) {                                                         \
            fids[n] = tmp_fid;                                                 \
            n++;                                                               \
            RETURN_IF_FULL();                                                  \
        }                                                                      \
    } while (0)

    varlist_t *candidates = NULL;
    hash512_t  hashtable;

    if (n != 0) {
        candidates = varlist_new(n, (m - n));
        int i, j;
        for (i = 0; i < n; i++) {
            hash512_clear(hashtable);
            int32_t fid = fids[i];
            int32_t gid = FID2GID(fid);
            int32_t meta = FID2GMETA(fid);
            if (gid == -2) {
                continue;
            }
            if (gid == -1) {
                varlist_push(candidates, i, meta);
                continue;
            }
            int     size = LOCAL_SIZE(gid);
            int32_t lid = meta, lid2;
            hash512_setbit(hashtable, lid);
            for (lid2 = 0; lid2 < size; lid2++) {
                int32_t lid3 = LOCAL_RECON(gid, lid, lid2);
                if (lid3 == -1 || hash512_hasbit(hashtable, lid3))
                    continue;
                hash512_setbit(hashtable, lid3);
                varlist_push(candidates, i, GMETA2FID(gid, lid3));
                if (varlist_is_row_full(candidates, i))
                    goto CONTINUE_OUTER;
            }
            lid2 = rand() % size;
            int max_size = min(candidates->ncols, size - 1);
            while (candidates->row_sizes[i] < max_size) {
                if (!hash512_hasbit(hashtable, lid2)) {
                    varlist_push(candidates, i, GMETA2FID(gid, lid2));
                    hash512_setbit(hashtable, lid2);
                }
                lid2 = (lid2 + 1) % size;
            }
        CONTINUE_OUTER:;
        }
        for (j = 0; j < candidates->ncols; j++) {
            for (i = 0; i < candidates->nrows; i++) {
                if (j >= candidates->row_sizes[i])
                    continue;
                PUSH_CONTEXT(varlist_at(candidates, i, j));
            }
        }
    }

    for (;;) {
        int32_t fid = rand() % (n_fids - 1) + 1;
        PUSH_CONTEXT(fid);
    }

EXIT:
    if (candidates != NULL)
        varlist_free(candidates);

#undef PUSH_CONTEXT
#undef RETURN_IF_FULL
}
