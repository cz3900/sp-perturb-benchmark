# Saunders b10 复现脚本 — SpatialProp vs CONCERT (PCC-delta)

在服务器 `~/spatial-pert/` 下跑(spatialprop / concert env)。dump 路径硬编为
`~/spatial-pert/outputs/{spatialprop,concert}/...`(见各脚本头)。顺序:

1. `concert_export_b10.py` — Saunders b10 → CONCERT `.h5`(counts_layer="X" + 确定性过滤)。
   CONCERT train 用 `--select_genes 0`(跳过内部 geneSelection,否则砍基因→丢细胞→size mismatch)。
2. `concert_gen_pertcells.py` — 写每个 guide 的 1-based `pert_cells_*.txt`(供 `--stage eval`)。
3. SpatialProp dump 由 `scripts/spatialprop/run_spatialprop.py` 跑出 `predicted_tempered`。
4. `score_b10_full.py` — 11 guide × {seed + niche},用 `run_benchmark`(自带 Gauss/GCN/null baseline)
   + `spbench.external.row_normalize`/`score_external_seed`。三者(观测/CONCERT/SpatialProp)行归一到
   同一 normalize_total 空间,只取 PCC-delta(scale-invariant)。
5. `plot_pcc_b10.py` — seed/niche 两个 heatmap → `outputs/figures/pcc_compare_b10.png`。
