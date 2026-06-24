# 空间扰动 benchmark — 实施路线图(plans 索引)

> 把 `benchmark_总设计_实现参考_2026-06-24.md` 切成可**一步步执行**的独立 plan。每个 plan 独立可测/可交付。
> 详细 TDD 展开策略:不依赖新数据的(Plan 2/3/4)可立即用 writing-plans 展开;**adapter(Plan 5-7)在执行前展开**(要先核实各自 csv/h5/mat 的列名与结构,现在写完整代码会有占位符,违反 no-placeholder)。

## 依赖顺序

```
Plan 1 (GCN 出图) ──┬─→ Plan 2 (退化模式) ──┐
                    ├─→ Plan 3 (perm null)  ├─→ Plan 5/6/7 (各 adapter) ─→ Plan 8 (rank+分层)
                    └─→ Plan 4 (end-to-end) ┘
```
Plan 2/3/4 可在 Plan 1 后并行;adapter 依赖 Plan 1(出图)+ 视情况 Plan 2(退化)。

---

## Plan 1 — GCN 跑通出图 ✅ 已写
- 文件:`2026-06-24-gcn-eval-pipeline.md`
- 范围:`compare_to_baseline`/`evaluate_seed` 暴露 per-repeat energy 分布;新 `plotting.py` 出 seed/niche 两张 box,GCN 具名为 `model+learned`。

## Plan 2 — 退化模式(无 cell_type / 无 NTC)
- **范围**:① 单一细胞系(Cheng、Binan-THP1/astro)`cell_type` 单值时,`control_reference_centers` 退化成全体 control(加判别测试确认);② 无 NTC(Shen,无 `'control'` 细胞)时兜底:control 池用 `perturbation=='none'` 的细胞。
- **改动点**:`spbench/reference_aggregate.py`(`control_reference_centers` 的 control-池定义 + 无-control 兜底分支);`aggregate_control` 同步;`harness._control_reference_aggregate` 同步。
- **验证**:synthetic 造"单一 cell_type"和"无 NTC"两种,`run_benchmark` 不崩 + Δseed/Δniche 非 NaN;判别测试:无 NTC 时 control 池 == `'none'` 细胞集。
- **依赖**:无(纯本地)。**可立即展开。**

## Plan 3 — permutation null(经验零分布 + p)
- **范围**:给每扰动的 Δseed/Δniche 一个置换零分布——随机抽同样多 non-perturbed 细胞当"假扰动 center",算其邻域 energy 分布;真 energy 相对该分布的经验 p。出图 dashed 可改用 permutation 的 95 分位(比 e_null 更严)。
- **改动点**:新 `spbench/permutation.py`(`permutation_null(data, perturbation, edges, n_perm, seed)` → null energy 列表);`config.run_benchmark` 可选 `n_perm` 算 p;`plotting` 接 p/分位线。
- **验证**:synthetic 上 inert 扰动 p 高(≈无效应)、planted 真扰动 p 低。
- **依赖**:Plan 1(energy 机制)。**可立即展开。**

## Plan 4 — end-to-end 模型 via `extra`(CONCERT 等)
- **范围**:定义"外部 niche 模型"接入路径——它的 niche 预测当 `compare_to_baseline(extra={name: niche_pred})` 传入(接口现成)、seed 预测走 `evaluate_seed`,出图里作为**独立一条(不经 GCN)**。先用 **mock end-to-end model** 测通整条路径,再接真 CONCERT 的 niche dump。
- **改动点**:新 `spbench/external.py`(`score_external_niche(data, perts, edges, niche_pred_fn, seed_pred_fn, name)`,复用 `fill_2x2` 的 niches + `compare_to_baseline(extra)`);`plotting.collect_prop_samples` 把 extra 方法纳入 boxes;`models/concert_model.py` 补 niche 输出(真接入时)。
- **验证**:mock model(niche = 观测 + 噪声)出现在 `res['compare'][p]['e_samples'][name]` 与 niche 图;它的 gain 介于 null 与 oracle 之间。
- **依赖**:Plan 1。CONCERT 真接入需先核实其 niche dump 格式(可后置)。**mock 版可立即展开。**

## Plan 5 — DhainautAdapter(spaceranger → spot StandardData)
- **范围**:读 `GSE193460_RAW`;**spot 当单元**;`X`=filtered.h5(32289)、`coords`=tissue_positions×scalefactors(µm)、`perturbation`=phenotypes 真 KO(去克隆后缀)、`is_control`=`KP_*`、`cell_type`=leiden_clusters、`batch`=4 GSM。字段映射见 `空间扰动benchmark_数据集适配记录` §4-§5。
- **验证**:`DhainautAdapter(...).load()` → StandardData;`run_benchmark`(KO=Tgfbr2/Ifngr2/Jak2,对照=KP)+ `plot_seed_prop` 跑通一张图。
- **依赖**:Plan 1(Dhainaut 有 cell_type,不需 Plan 2)。**执行前展开**(核实 spaceranger h5 + spot_annotation.csv 列名)。

## Plan 6 — BinanTumorsAdapter(finaltables CSV)
- **范围**:读 tumors finaltables;`X`=merfishcounttable(550)、`gene_names`=gene_mapping、`coords`=index_xy、`perturbation`=allcellsPerturbationTable、`cell_type`=tcells/tumorcells 分组、`meta`=withimmuneneighbor。
- **验证**:load → run_benchmark + 出图;cell_type 含 T/肿瘤。
- **依赖**:Plan 1。**执行前展开**(核实各 CSV 列名/对齐方式)。

## Plan 7 — ChengAdapter(.mat + codebook 恢复基因名)
- **范围**:`scipy.io.loadmat` 读 `CellList_PerturbRaeFISH.mat` + `Codebook_MERFISH.mat`(ShortName↔Code 恢复 `gene_names`);`perturbation`=Top1Target、`cell_type`=`'A549'`(退化)、`batch`=DataSet。
- **验证**:load → `gene_names` 是基因名(非 barcode)、`X` 列与之对齐;run_benchmark 在退化模式跑通。
- **依赖**:Plan 1 + **Plan 2(退化模式)**。**执行前展开**(核实 .mat struct 字段)。

## Plan 8(可选) — ShenAdapter + 跨数据集 rank + MC-spatial 分层
- **范围**:`ShenAdapter`(全转录组,无 NTC → 用 Plan 2 兜底,n 小报告局限);跨数据集 normalized-gain + rank 聚合(不对齐表征);MC-spatial 象限分层(复用现有 `mc_spatial_join`/`mc_spatial_report`)。
- **依赖**:Plan 1 + 2。

---

## 执行方式
逐个 plan、subagent-driven(每个 task 一个 **Opus** 子代理实现 + 两段式审查)。下一步:执行 Plan 1 → 展开并执行 Plan 2/3/4(不依赖数据)→ 逐个展开并执行 adapter(执行前核实数据格式)。
