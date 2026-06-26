"""SPAC-seq (PKU ZengLab, Visium HD spatial CRISPR screen) cell-level preprocessing.

Two resolutions must be linked:
- guide UMIs live on **8um bins** (``filtered_guide_bc_matrix.h5`` -- AnnData despite the .h5 ext,
  CSR bins x 1520 guides);
- expression + cell types live on **segmented cells** (``segmentation/filtered_feature_cell_matrix.h5``
  10x v3 matrix, ``cell_annotations.json``, and ``segmentation/*.geojson`` polygons).

bin -> cell mapping (validated subQ-1: 93.8% of in_tissue bins land in a cell):
  The geojson polygons are full-res-pixel coordinates but **translated** (segmentation ran on a
  left-cropped image). ``tissue_positions.parquet`` (from the Spaceranger raw output, staged under
  ``spatial_meta/<sample>/``) gives each bin's ``pxl_row/col_in_fullres``. Scale already matches in
  full-res px, so we estimate the per-sample translation by centroid alignment (pure translation),
  then point-in-polygon (shapely STRtree) assigns each bin to a cell_id. ~6% of bins fall in
  intercellular space (cell_id = -1) -> their guides are dropped (default) or sent to the nearest cell.

Guide UMIs are summed per cell, the two sgRNA per gene collapsed, then per-cell calling labels each
cell into the StandardData convention: single dominant target gene -> gene KO; only sgnon-targeting
-> 'control'; nothing detected / ambiguous multiplet -> 'none' (control-pool fallback, Plan 2).

The heavy geometry runs once on the server (see scripts there); this module holds the reusable,
testable logic. ``SpacSeqAdapter`` (spac_seq.py) just reads the resulting processed .h5ad.
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd, h5py
from scipy.sparse import csr_matrix

from ..data import StandardData, CONTROL, UNLABELED

NON_TARGETING = "non-targeting"


# --------------------------------------------------------------------------- IO
def read_anndata_h5(path):
    """Read an AnnData-format .h5 (X as CSR + obs/_index + var/_index). The SPAC-seq transcriptome
    and perturbation files use this layout despite the .h5 extension. Returns (X_csr, obs_idx, var_idx)."""
    with h5py.File(path, "r") as f:
        X = csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                       shape=(f["obs/_index"].shape[0], f["var/_index"].shape[0]))
        obs = _decode(f["obs/_index"][:])
        var = _decode(f["var/_index"][:])
    return X, obs, var


def load_cell_expression(sample_dir):
    """Cell x gene expression from the segmentation matrix. Returns AnnData (obs_names = 'cellid_*-1')."""
    import scanpy as sc
    ad = sc.read_10x_h5(os.path.join(sample_dir, "segmentation", "filtered_feature_cell_matrix.h5"))
    ad.var_names_make_unique()
    return ad


def load_cell_types(sample_dir):
    """{cellid_*-1: celltype} from cell_annotations.json (1:1 with segmentation cells)."""
    with open(os.path.join(sample_dir, "cell_annotations.json")) as fh:
        return json.load(fh)


def load_cell_polygons(sample_dir, which="cell_segmentations"):
    """Read geojson polygons. Returns (cell_ids[int], polys[list of (k,2) arrays], centroids[(n,2)])."""
    with open(os.path.join(sample_dir, "segmentation", f"{which}.geojson")) as fh:
        gj = json.load(fh)
    cids, polys, cents = [], [], []
    for feat in gj["features"]:
        a = np.asarray(feat["geometry"]["coordinates"][0], dtype=float)
        cids.append(int(feat["properties"]["cell_id"]))
        polys.append(a)
        cents.append(a.mean(axis=0))
    return np.asarray(cids, dtype=np.int64), polys, np.asarray(cents, dtype=float)


def load_bin_pixel_coords(spatial_meta_dir):
    """in_tissue bins with full-res pixel coords. Returns DataFrame indexed by bin barcode with
    columns [x, y] = [pxl_col_in_fullres, pxl_row_in_fullres] (x=col, y=row to match geojson)."""
    tp = pd.read_parquet(os.path.join(spatial_meta_dir, "tissue_positions.parquet"))
    tp = tp[tp["in_tissue"] == 1]
    out = pd.DataFrame({"x": tp["pxl_col_in_fullres"].to_numpy(),
                        "y": tp["pxl_row_in_fullres"].to_numpy()},
                       index=tp["barcode"].to_numpy())
    return out


def cellbarcode_to_id(bc):
    """'cellid_000000001-1' -> 1 (matches geojson properties.cell_id)."""
    return int(str(bc).split("_", 1)[1].split("-", 1)[0])


def _decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


# ------------------------------------------------------------------ bin -> cell
def estimate_offset(bin_xy, cell_centroids):
    """Per-sample translation aligning bin full-res-pixel cloud to the (cropped) geojson frame.
    Pure translation: scale already matches in full-res px. offset = mean(bin) - mean(cell)."""
    return bin_xy.mean(axis=0) - cell_centroids.mean(axis=0)


def cell_radii(polys, cell_centroids):
    """Per-cell radius = max vertex distance from its centroid (containment proxy for the KNN gate)."""
    return np.array([np.sqrt(((p - c) ** 2).sum(axis=1)).max() for p, c in zip(polys, cell_centroids)])


def assign_bins_to_cells(bin_xy, cell_ids, polys, cell_centroids=None, offset=None,
                         method="knn", radii=None, radius_scale=1.0):
    """Assign each bin to a cell_id (-1 if intercellular). Returns (bin_cellid[int64], offset).

    bin_xy: (n_bins, 2) full-res pixel [x, y].
    method='knn' (default): nearest cell centroid via scipy cKDTree, gated by the cell's radius
      (assign only if dist <= radius_scale * radius[nearest]). O(n log n) -- seconds for ~600k bins;
      densely-tiled cells make this ~equivalent to point-in-polygon (validated containment ~94%). The
      exact shapely point-in-polygon path (method='polygon') is correct but builds ~300k Polygons and
      is far slower; kept for spot-checking.
    """
    if cell_centroids is None:
        cell_centroids = np.array([p.mean(axis=0) for p in polys])
    if offset is None:
        offset = estimate_offset(bin_xy, cell_centroids)
    bxy = bin_xy - offset
    cids = np.asarray(cell_ids)

    if method == "knn":
        from scipy.spatial import cKDTree
        if radii is None:
            radii = cell_radii(polys, cell_centroids)
        dist, idx = cKDTree(cell_centroids).query(bxy, k=1)
        ok = dist <= radius_scale * radii[idx]
        return np.where(ok, cids[idx], -1).astype(np.int64), offset

    import shapely
    from shapely.geometry import Polygon
    from shapely import STRtree
    pts = shapely.points(bxy[:, 0], bxy[:, 1])
    bin_idx, poly_idx = STRtree([Polygon(p) for p in polys]).query(pts, predicate="contains")
    out = np.full(len(bxy), -1, dtype=np.int64)
    out[bin_idx[::-1]] = cids[poly_idx[::-1]]   # first match wins
    return out, offset


def aggregate_guides_to_cells(bin_guide, bin_cellid, cell_id_order):
    """Sum bin guide counts into cells. bin_guide: (n_bins, n_guides) CSR aligned to bin order.
    bin_cellid: cell_id per bin (-1 dropped). cell_id_order: cell_ids defining output rows.
    Returns (n_cells, n_guides) CSR."""
    pos = {int(c): i for i, c in enumerate(cell_id_order)}
    rows = np.fromiter((pos.get(int(c), -1) for c in bin_cellid), dtype=np.int64, count=len(bin_cellid))
    keep = np.where(rows >= 0)[0]
    A = csr_matrix((np.ones(keep.size, dtype=np.float32), (rows[keep], keep)),
                   shape=(len(cell_id_order), bin_guide.shape[0]))
    return (A @ bin_guide).tocsr()


# --------------------------------------------------------------- guide calling
def guide_target(name):
    """'sgTgfbr2_1' -> 'Tgfbr2'; 'sgH2-M10.1_2' -> 'H2-M10.1'; 'sgnon-targeting_7' -> 'non-targeting'."""
    s = name[2:] if name.startswith("sg") else name
    base = s.rsplit("_", 1)[0] if "_" in s else s
    return NON_TARGETING if base.startswith(NON_TARGETING) else base


def collapse_to_genes(cell_guide, guide_names):
    """Collapse the two sgRNA per gene. Returns (cell x gene CSR, gene_list). 'non-targeting' is one column."""
    targets = np.array([guide_target(g) for g in guide_names])
    genes = sorted(set(targets), key=lambda g: (g == NON_TARGETING, g))  # non-targeting last
    gidx = {g: i for i, g in enumerate(genes)}
    cols = np.array([gidx[t] for t in targets])
    n_cells = cell_guide.shape[0]
    M = csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)), shape=(len(cols), len(genes)))
    return (cell_guide @ M).tocsr(), genes


def call_perturbation(cell_gene, genes, min_umi=2, min_dominance=2.0):
    """Per-cell label from collapsed gene-level guide counts.

    min_umi: top gene must reach this UMI count to call a perturbation.
    min_dominance: top / second-highest ratio required for a clean singlet (else 'none' = multiplet).

    Returns dict with 'perturbation' (StandardData labels) and QC arrays (top_umi, n_genes_detected,
    dominance). Labels: gene KO name | CONTROL (non-targeting only) | UNLABELED ('none': nothing
    detected or ambiguous multiplet -> control pool fallback, Plan 2)."""
    X = cell_gene.toarray() if hasattr(cell_gene, "toarray") else np.asarray(cell_gene)
    genes = np.asarray(genes)
    nt_col = int(np.where(genes == NON_TARGETING)[0][0]) if NON_TARGETING in genes else -1

    order = np.argsort(-X, axis=1)
    top_i = order[:, 0]
    top_v = X[np.arange(X.shape[0]), top_i]
    second_v = X[np.arange(X.shape[0]), order[:, 1]] if X.shape[1] > 1 else np.zeros(X.shape[0])
    n_det = (X > 0).sum(axis=1)
    dominance = np.where(second_v > 0, top_v / np.maximum(second_v, 1e-9), np.inf)

    labels = np.full(X.shape[0], UNLABELED, dtype=object)
    called = (top_v >= min_umi) & (dominance >= min_dominance)
    for i in np.where(called)[0]:
        labels[i] = CONTROL if top_i[i] == nt_col else str(genes[top_i[i]])
    return {
        "perturbation": labels.astype(str),
        "top_umi": top_v, "second_umi": second_v,
        "n_genes_detected": n_det, "dominance": dominance,
        "genes": genes,
    }


# ---------------------------------------------------------------- full builder
def build_sample(sample_dir, spatial_meta_dir, sample_name, offset=None,
                 min_umi=2, min_dominance=2.0, return_qc=False):
    """End-to-end one sample -> per-cell arrays. Returns dict (expression AnnData, coords, perturbation,
    cell_type, plus QC). Assemble across samples into StandardData with `to_standard_data`."""
    ad = load_cell_expression(sample_dir)
    cell_bc = np.asarray(ad.obs_names)
    cell_ids = np.array([cellbarcode_to_id(b) for b in cell_bc], dtype=np.int64)

    types = load_cell_types(sample_dir)
    cell_type = np.array([types.get(b, "Others") for b in cell_bc])

    poly_ids, polys, cents = load_cell_polygons(sample_dir)
    # cell centroid coords (geojson frame) keyed by cell_id, aligned to expression cell order
    cent_by_id = {int(c): cents[k] for k, c in enumerate(poly_ids)}
    coords = np.array([cent_by_id.get(int(c), [np.nan, np.nan]) for c in cell_ids], dtype=float)

    bins = load_bin_pixel_coords(spatial_meta_dir)
    bin_guide, bin_bc, guide_names = read_anndata_h5(os.path.join(sample_dir, "filtered_guide_bc_matrix.h5"))
    # align guide bins to parquet pixel coords (intersection, preserve guide-matrix row order)
    keep = np.isin(bin_bc, bins.index.to_numpy())
    bin_guide, bin_bc = bin_guide[keep], bin_bc[keep]
    bin_xy = bins.loc[bin_bc, ["x", "y"]].to_numpy()

    bin_cellid, offset = assign_bins_to_cells(bin_xy, poly_ids, polys, cents, offset=offset)
    cell_guide = aggregate_guides_to_cells(bin_guide, bin_cellid, cell_ids)
    cell_gene, genes = collapse_to_genes(cell_guide, guide_names)
    call = call_perturbation(cell_gene, genes, min_umi=min_umi, min_dominance=min_dominance)

    out = dict(adata=ad, coords=coords, cell_type=cell_type, sample=sample_name,
               perturbation=call["perturbation"], offset=offset,
               n_bins_assigned=int((bin_cellid >= 0).sum()), n_bins=len(bin_cellid),
               guide_names=guide_names, genes=genes)
    if return_qc:
        out["qc"] = {k: call[k] for k in ("top_umi", "second_umi", "n_genes_detected", "dominance")}
        out["cell_guide"] = cell_guide
    return out


def to_standard_data(samples, meta_name="SPAC-seq"):
    """Stack per-sample build_sample() outputs into one cell-level StandardData."""
    import anndata as adlib
    ads = [s["adata"] for s in samples]
    var_common = sorted(set.intersection(*[set(a.var_names) for a in ads]))
    X = np.vstack([a[:, var_common].X.toarray() if hasattr(a.X, "toarray") else a[:, var_common].X
                   for a in ads]).astype(np.float32)
    coords = np.vstack([s["coords"] for s in samples])
    pert = np.concatenate([s["perturbation"] for s in samples]).astype(str)
    ctype = np.concatenate([s["cell_type"] for s in samples]).astype(str)
    batch = np.concatenate([np.full(len(s["perturbation"]), s["sample"]) for s in samples]).astype(str)
    return StandardData(X=X, coords=coords, perturbation=pert, cell_type=ctype, batch=batch,
                        gene_names=list(var_common), meta={"name": meta_name})
