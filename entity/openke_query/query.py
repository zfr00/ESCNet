from collections import defaultdict
from glob import glob
import math
import os.path as osp
import sys
import pyximport
import numpy as np
import re

pyximport.install(
    setup_args={
        "include_dirs": [
            np.get_include(),
            osp.abspath(osp.dirname(__file__)),
            "/opt/conda/include/python3.7m", ### replace /opt/conda/ to your path
        ]
    },
    reload_support=True,
    inplace=True,
    language_level=3,
)


sys.path.append('./your_path/entity/openke_query')   ### replace your_path
import query_cpy as qcore

mid2fid = {}
term2fid = {}
term2mid = {}
fid2term = defaultdict(set)

DATA_DIR = "./your_path/entity/"   ### replace your_path


def build_dict(data_dir=DATA_DIR):
    print("building dict...")
    global mid2fid, term2fid, term2mid
    if len(mid2fid):
        return
    with open(osp.join(data_dir, "fid2mid2name.txt")) as fd:
        for line in fd:
            line = line.strip("\n")
            if not line:
                continue
            fid, mid, term = line.split("\t")
            fid = int(fid)
            mid2fid[mid] = fid
            term2fid[term] = fid
            term2mid[term] = mid
            fid2term[fid].add(term)

        terms = set(term2fid)
        for term in terms:
            if not term.endswith(")"):  # do a fast check
                continue
            grp = re.match(r"^(.+?)\s*\(.*\)$", term)
            if grp is None:
                continue
            stripped_term = grp.group(1)
            if stripped_term not in terms:
                fid = term2fid[term]
                mid = term2mid[term]
                term2fid[stripped_term] = fid
                term2mid[stripped_term] = mid
                fid2term[fid].add(stripped_term)

    print("dict built!")


def build_index(data_dir=DATA_DIR):
    print("Building index...")
    major_edges = np.memmap(
        osp.join(data_dir, "confuse", "g_major.bin"), mode="r", dtype=np.int32
    )

    global_recon = np.memmap(
        osp.join(data_dir, "recon", "g_major.bin"), mode="r", dtype=np.int16
    )

    local_recons = np.memmap(
        osp.join(data_dir, "recon", "locals.bin"), mode="r", dtype=np.int16
    )

    fid2gmeta = np.memmap(osp.join(data_dir, "fid2gmeta.bin"), mode="r", dtype=np.int32)

    major_minor_edges = np.memmap(
        osp.join(data_dir, "confuse", "major_minor_edges.bin"), mode="r", dtype=np.int32
    )

    qcore.setup_index(
        major_edges, global_recon, local_recons, fid2gmeta, major_minor_edges
    )
    print("Index built!")


query_pairs = qcore.query_pairs

_embedding = None


def get_embedding():
    global _embedding
    if _embedding is not None:
        return _embedding
    _embedding = np.memmap(
        osp.join(DATA_DIR, "embedding.bin"), mode="r", dtype=np.float32
    ).reshape(-1, 50)
    return _embedding


def pair_embedding(fids, max_hop, alpha=0.9, beta=0.9):
    """
    fids: N x 2
    """
    N = fids.shape[0]
    fids = (
        np.concatenate([fids, fids[:, ::-1]], axis=-1).reshape(-1, 2).astype(np.int32)
    )
    paths, _ = query_pairs(fids, max_hop)  # 2N x max_hop
    emb = get_embedding()[paths]  # 2N x max_hop x 50
    coef = alpha * beta ** np.arange(-1, max_hop - 1)
    coef[0] = 1
    coef = coef.reshape(1, -1, 1).repeat(2 * N, axis=0)
    coef[paths == 0] = 0
    emb = (emb * coef).sum(axis=1) / (coef.sum(axis=1))  # 2N x 50
    return emb.reshape(N, 2, -1)


def cross_modal_embedding(t_fids, v_fids, k, max_hop, alpha=0.9, beta=0.9):
    """
    Args:
        t_fids: List[int]. List of fids for textual entities. May be empty.
        v_fids: List[int]. List of fids for visual entities. May be empty.
        k: int. Each set will be enlarged to contain no less than k elements.
        max_hop: int. Max steps to solve shortest path.
        alpha, beta: float. Parameters for aggregating embeddings.

    This function does the following:
        1. Enlarge t_fids to no less than k elements. Denote the size as nt;
        2. Enlarge v_fids to no less than k elements. Denote the size as nv;
        3. Calculate the T-T, T-V, V-V embeddings.

    Returns: (tt_emb, tv_emb, vv_emb), whose shapes are
        tt_emb: C(nt, 2) x 2 x 50
        tv_emb: (nt * nv) x 2 x 50
        vv_emb: C(nv, 2) x 2 x 50
    """
    t_fids = qcore.enlarge_context(np.array(t_fids, dtype=np.int32), k)
    v_fids = qcore.enlarge_context(np.array(v_fids, dtype=np.int32), k)

    nt = t_fids.shape[0]
    nv = v_fids.shape[0]

    tt_indices = np.stack(np.triu_indices(nt, 1), axis=-1)
    tt_fids = t_fids[tt_indices]
    tt_emb = pair_embedding(tt_fids, max_hop, alpha, beta)

    tv_indices = np.indices((nt, nv)).reshape(2, -1)
    tv_fids = np.stack([t_fids[tv_indices[0]], v_fids[tv_indices[1]]], axis=-1)
    tv_emb = pair_embedding(tv_fids, max_hop, alpha, beta)

    vv_indices = np.stack(np.triu_indices(nv, 1), axis=-1)
    vv_fids = v_fids[vv_indices]
    vv_emb = pair_embedding(vv_fids, max_hop, alpha, beta)

    return tt_emb, tv_emb, vv_emb


if __name__ == "__main__":
    build_index()
    # cross_modal_embedding([42], [45, 128, 120], 3, 20)
    tt,tv,vv = cross_modal_embedding([42], [45, 128, 120], 3, 20)
    # print(type(tt))
    build_dict()
    print(term2fid["Americans"])
    t_fids = qcore.enlarge_context(np.array([45,42,120], dtype=np.int32), 5)
    # breakpoint()
    print(t_fids)
    embedding = get_embedding()
    print(embedding[t_fids])
