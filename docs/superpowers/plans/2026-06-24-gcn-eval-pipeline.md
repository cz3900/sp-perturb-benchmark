# GCN 跑通评测流程(seed/prop 对比图) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `run_benchmark` 的输出能画成"每数据集 seed + niche 两张模型对比 box plot"(对标 CONCERT 图),先用 GCN(learned prop)作为具名被评测方法,在 synthetic + Saunders 上端到端跑通。

**Architecture:** 现有 `run_benchmark` 已经为每个扰动算出每个方法(2x2 四格 + null + oracle)的 matched-n energy 分布,但 `compare_to_baseline` 只返回均值、`evaluate_seed` 只返回单值 PCC/MSE。本计划(1)把这些**per-repeat energy 分布**暴露出来当 box plot 数据,(2)新增 `spbench/plotting.py` 从 `res` 提取各方法分布并画 seed/prop 两张 box plot。GCN 作为 `model+learned` 方法,在 niche 图里具名显示为 "GCN"。

**Tech Stack:** Python 3.11, numpy, matplotlib(Agg), pytest;复用现有 `spbench`(compare/harness/config/models/metrics)。

---

## File Structure

- `spbench/compare.py`(改):`compare_to_baseline` 返回加 `e_samples`(niche 每方法 per-repeat 分布);`evaluate_seed` 加 `repeats/seed/max_n` 参数与 `e_samples`(seed 的 model/null per-repeat 分布)。
- `spbench/plotting.py`(新):`collect_prop_samples` / `collect_seed_samples`(从 `res` 提取 `{方法标签: 分布}` + dashed 基线)、`plot_seed_prop`(画两张 box)。GCN 的标签映射在这里。
- `tests/test_compare_samples.py`(新):测两个 `e_samples`。
- `tests/test_plotting.py`(新):测 collect + plot。
- `scripts/plot_methods_demo.py`(新):端到端 demo(synthetic → 出图存 png)。

约定:测试用 `tests/conftest.py` 已有的 `synth` fixture(`make_synthetic(seed=0)`,扰动名 `"P0"`)。GCN 测试一律用小参数 `gcn_kwargs={"hidden": 16, "epochs": 5}` 提速。

---

### Task 1: `compare_to_baseline` 暴露 niche per-repeat 分布

**Files:**
- Modify: `spbench/compare.py:111`
- Test: `tests/test_compare_samples.py`(新建)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_compare_samples.py`:

```python
import numpy as np
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2, _control_reference_aggregate, _control_residuals
from spbench.compare import compare_to_baseline
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN


def _niches(synth):
    edges = build_knn_graph(synth, k=8)
    X_ref = _control_reference_aggregate(synth, edges)
    resid = _control_residuals(synth)
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=5).fit(synth, edges)
    g = fill_2x2(synth, "P0", edges, seed, base, learned, k_ref=5, X_ref=X_ref,
                 return_niches=True, residuals=resid)
    return g["_niches"], resid


def test_compare_returns_per_repeat_samples(synth):
    niches, resid = _niches(synth)
    out = compare_to_baseline(niches, residuals=resid, repeats=20)
    assert "e_samples" in out
    assert "model+learned" in out["e_samples"], "GCN (learned prop) must be a named method"
    for k, mean_e in out["e"].items():
        s = out["e_samples"][k]
        assert len(s) == 20
        assert np.isclose(np.nanmean(s), mean_e)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/cz/Documents/ZengLab/model/sp-perturb-benchmark && python -m pytest tests/test_compare_samples.py -q`
Expected: FAIL — `KeyError: 'e_samples'` 或 `assert "e_samples" in out`。

- [ ] **Step 3: 最小实现**

把 `spbench/compare.py:111` 的 return 改为(在末尾加 `e_samples`):

```python
    return {"e": e, "gain": gain, "pcc": pcc, "n": n, "has_effect": has_effect,
            "e_samples": {k: list(v) for k, v in acc.items()}}
```

(`acc` 已在该函数行 92-96 算好,是 `{method: [per-repeat energy]}`,这里只是返回它。)

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_compare_samples.py::test_compare_returns_per_repeat_samples -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add spbench/compare.py tests/test_compare_samples.py
git commit -m "feat(compare): expose per-repeat energy samples (niche box-plot data)"
```

---

### Task 2: `evaluate_seed` 暴露 seed per-repeat 分布

**Files:**
- Modify: `spbench/compare.py:44-59`
- Test: `tests/test_compare_samples.py`(追加)

- [ ] **Step 1: 写失败测试**

在 `tests/test_compare_samples.py` 末尾追加:

```python
from spbench.compare import evaluate_seed


def test_evaluate_seed_returns_energy_samples(synth):
    niches, _ = _niches(synth)
    out = evaluate_seed(niches, repeats=15)
    assert "e_samples" in out
    assert set(out["e_samples"]) >= {"model", "null"}
    assert len(out["e_samples"]["model"]) == 15
    assert all(s >= 0 for s in out["e_samples"]["model"])
    # 旧字段仍在
    assert "pcc_delta" in out and "mse" in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_compare_samples.py::test_evaluate_seed_returns_energy_samples -q`
Expected: FAIL — `TypeError: evaluate_seed() got an unexpected keyword 'repeats'` 或 `assert "e_samples" in out`。

- [ ] **Step 3: 最小实现**

把 `spbench/compare.py` 的整个 `evaluate_seed`(行 44-59)替换为:

```python
def evaluate_seed(niches, eval_X=None, repeats=20, seed=0, max_n=300):
    """Direct seed score: MODEL seed vs observed perturbed cells. Returns PCC-delta (direction,
    baseline=matched control), MSE (magnitude), AND `e_samples` = per-repeat matched-n energy
    distance for the model seed and the null (control seed) — the seed box-plot data source.

    eval_X (callable | None): unified scoring-space transform applied to pred/obs/ref before
    scoring. repeats/seed/max_n control the matched-n energy resampling for `e_samples`."""
    obs = _apply_eval_X(niches.get("seed_obs", np.zeros((0, 0))), eval_X)
    pred = _apply_eval_X(niches.get("seed_pred", np.zeros((0, 0))), eval_X)
    ref = _apply_eval_X(niches.get("seed_ref", np.zeros((0, 0))), eval_X)
    if len(obs) == 0 or len(pred) == 0 or len(ref) == 0:
        return {"pcc_delta": float("nan"), "mse": float("nan"), "n": int(len(obs)),
                "e_samples": {}}
    clouds = {"model": pred, "null": ref}
    n = max(2, min(len(obs), len(pred), len(ref), max_n))
    rng = np.random.default_rng(seed + 1)
    samp = {k: [] for k in clouds}
    for _ in range(repeats):
        O = _sub(obs, n, rng)
        for k, c in clouds.items():
            samp[k].append(energy_distance(_sub(c, n, rng), O))
    return {"pcc_delta": get_metric("pcc_delta").compute(pred, obs, {"reference": ref}),
            "mse": get_metric("mse").compute(pred, obs), "n": int(len(obs)),
            "e_samples": {k: list(v) for k, v in samp.items()}}
```

(`energy_distance`、`_sub`、`get_metric`、`_apply_eval_X` 都已在 `compare.py` 顶部/同文件可用。)

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_compare_samples.py -q`
Expected: PASS(两个测试都过)

- [ ] **Step 5: 提交**

```bash
git add spbench/compare.py tests/test_compare_samples.py
git commit -m "feat(compare): evaluate_seed exposes per-repeat energy samples (seed box-plot data)"
```

---

### Task 3: `plotting.collect_*` 从 res 提取各方法分布(GCN 具名)

**Files:**
- Create: `spbench/plotting.py`
- Test: `tests/test_plotting.py`(新建)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_plotting.py`:

```python
import numpy as np
from spbench.config import run_benchmark
from spbench.plotting import collect_prop_samples, collect_seed_samples

GCN_KW = {"hidden": 16, "epochs": 5}


def test_collect_prop_samples_has_named_gcn(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_prop_samples(res)
    assert "GCN" in boxes, "learned prop must show up named 'GCN'"
    assert "Gaussian" in boxes
    assert isinstance(boxes["GCN"], np.ndarray) and boxes["GCN"].size > 0
    assert "null" in dashed and "oracle" in dashed


def test_collect_seed_samples(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_seed_samples(res)
    assert len(boxes) >= 1
    assert "null" in dashed
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_plotting.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'spbench.plotting'`。

- [ ] **Step 3: 最小实现**

创建 `spbench/plotting.py`(只到 collect 这一层,plot 在 Task 4):

```python
"""Turn run_benchmark output into per-dataset seed/niche method-comparison box plots.

Each method's box = its pooled per-repeat matched-n energy distances (from `e_samples`).
GCN is the learned-prop method (`model+learned`); shown named "GCN" in the niche plot.
Dashed lines = no-effect null and oracle ceiling (mean energy)."""
import numpy as np

# 2x2 method key -> niche-plot label. GCN = learned prop (model seed + GCN); Gaussian = baseline prop.
PROP_LABELS = {"model+base": "Gaussian", "model+learned": "GCN",
               "GT+base": "Gaussian (GT seed)", "GT+learned": "GCN (GT seed)"}


def collect_prop_samples(res, box_methods=("model+base", "model+learned"),
                         dashed_methods=("null", "oracle")):
    """{label: pooled per-repeat energy array} for the box methods, plus {name: mean energy}
    for the dashed baselines. Pools across all perturbations in res['compare']."""
    boxes, dashed = {}, {}
    cmp = res.get("compare", {})
    for m in box_methods:
        pooled = []
        for c in cmp.values():
            pooled += list(c.get("e_samples", {}).get(m, []))
        if pooled:
            boxes[PROP_LABELS.get(m, m)] = np.asarray(pooled, float)
    for m in dashed_methods:
        vals = [c["e"][m] for c in cmp.values() if m in c.get("e", {})]
        if vals:
            dashed[m] = float(np.nanmean(vals))
    return boxes, dashed


def collect_seed_samples(res, model_label="seed model"):
    """{label: pooled per-repeat energy} for the model seed, plus {'null': mean} dashed."""
    boxes, dashed = {}, {}
    seed = res.get("seed", {})
    pooled = [s for v in seed.values() for s in v.get("e_samples", {}).get("model", [])]
    if pooled:
        boxes[model_label] = np.asarray(pooled, float)
    null_p = [s for v in seed.values() for s in v.get("e_samples", {}).get("null", [])]
    if null_p:
        dashed["null"] = float(np.nanmean(null_p))
    return boxes, dashed
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_plotting.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add spbench/plotting.py tests/test_plotting.py
git commit -m "feat(plotting): collect per-method energy distributions from res (GCN named)"
```

---

### Task 4: `plotting.plot_seed_prop` 画 seed/niche 两张 box plot

**Files:**
- Modify: `spbench/plotting.py`(追加绘图函数)
- Test: `tests/test_plotting.py`(追加)

- [ ] **Step 1: 写失败测试**

在 `tests/test_plotting.py` 末尾追加:

```python
import matplotlib
matplotlib.use("Agg")
from spbench.plotting import plot_seed_prop


def test_plot_seed_prop_returns_two_axes_with_gcn(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    fig = plot_seed_prop(res)
    assert fig is not None
    assert len(fig.axes) == 2
    niche_labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert "GCN" in niche_labels
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_plotting.py::test_plot_seed_prop_returns_two_axes_with_gcn -q`
Expected: FAIL — `ImportError: cannot import name 'plot_seed_prop'`。

- [ ] **Step 3: 最小实现**

在 `spbench/plotting.py` 末尾追加:

```python
def _draw_boxes(ax, boxes, dashed, ylabel, title):
    import numpy as np
    labels = list(boxes)
    data = [np.log(np.clip(boxes[l], 1e-9, None)) for l in labels]
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    _dash_colors = {"null": "#888888", "oracle": "#1d9e75"}
    for name, val in dashed.items():
        ax.axhline(np.log(max(val, 1e-9)), ls="--", lw=1.2,
                   color=_dash_colors.get(name, "#888888"), label=name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if dashed:
        ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", rotation=30)


def plot_seed_prop(res, figsize=(11, 4.2)):
    """One figure, two box plots: Δseed (left) and Δniche (right). x = methods (GCN named),
    box = pooled per-repeat log energy distance, dashed = null / oracle baselines."""
    import matplotlib.pyplot as plt
    sb, sd = collect_seed_samples(res)
    pb, pd = collect_prop_samples(res)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _draw_boxes(ax1, sb, sd, "log E-distance", "seed (D1)")
    _draw_boxes(ax2, pb, pd, "log E-distance", "niche (D2)")
    fig.tight_layout()
    return fig
```

(`tick_labels` 是 matplotlib ≥3.9 的参数;若环境报 `unexpected keyword 'tick_labels'`,把它改成 `labels`。)

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_plotting.py -q`
Expected: PASS(三个测试都过)

- [ ] **Step 5: 提交**

```bash
git add spbench/plotting.py tests/test_plotting.py
git commit -m "feat(plotting): plot_seed_prop draws seed/niche method-comparison box plots"
```

---

### Task 5: 端到端 demo + 全套回归

**Files:**
- Create: `scripts/plot_methods_demo.py`
- Test: `tests/test_plotting.py`(追加 smoke)

- [ ] **Step 1: 写失败测试**

在 `tests/test_plotting.py` 末尾追加:

```python
def test_demo_script_writes_png(tmp_path, synth):
    from spbench.plotting import plot_seed_prop
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    fig = plot_seed_prop(res)
    out = tmp_path / "methods.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: 跑测试确认失败(或直接通过)**

Run: `python -m pytest tests/test_plotting.py::test_demo_script_writes_png -q`
Expected: PASS(此测试只用已实现的 API;若失败,说明 Task 4 的 fig 不可保存,需修)。先跑确认绿,再写 demo 脚本。

- [ ] **Step 3: 写 demo 脚本**

创建 `scripts/plot_methods_demo.py`:

```python
"""Demo: run the benchmark on synthetic data and save the seed/niche method-comparison plot.
Real datasets: replace make_synthetic with an Adapter().load() and a real perturbation list."""
import argparse
import matplotlib
matplotlib.use("Agg")
from spbench.synthetic import make_synthetic
from spbench.config import run_benchmark
from spbench.plotting import plot_seed_prop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_demo.png")
    args = ap.parse_args()
    data = make_synthetic(seed=0)
    res = run_benchmark(data, perturbations=["P0"], gcn_kwargs={"hidden": 32, "epochs": 20},
                        progress=False)
    fig = plot_seed_prop(res)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑脚本 + 全套回归**

Run: `python scripts/plot_methods_demo.py --out /tmp/methods_demo.png && python -m pytest -q`
Expected: 打印 `wrote /tmp/methods_demo.png`;pytest 全绿(原有 149 + 新增,无回归)。

- [ ] **Step 5: 提交**

```bash
git add scripts/plot_methods_demo.py tests/test_plotting.py
git commit -m "feat(plotting): end-to-end demo script + smoke test (GCN pipeline runs)"
```

---

## Self-Review

- **Spec 覆盖**:对标总设计 §5 出图规格——每数据集 seed+niche 两张 box、模型横排、box=重复、dashed=null/oracle、GCN 具名为 learned prop:Task 1(niche 数据)+ Task 2(seed 数据)+ Task 3(GCN 标签)+ Task 4(两张图)+ Task 5(端到端)覆盖。**本计划不含**:permutation null、end-to-end via extra(CONCERT)、5 个新 adapter、退化模式、跨数据集 rank——这些是后续 plan(总设计 §7 后续步骤),本计划只"用 GCN 跑通流程"。
- **Placeholder 扫描**:无 TBD/TODO;每个 code step 都有完整代码。
- **类型一致**:`e_samples` 在 Task 1(niche,键=2x2 方法名)与 Task 2(seed,键 model/null)结构一致(都是 `{str: [float]}`);Task 3 的 `collect_*` 读这两处;`PROP_LABELS["model+learned"]=="GCN"` 与 Task 4 测试断言的 `"GCN"` 一致。
- **真实接口**:`fill_2x2`/`_control_reference_aggregate`/`_control_residuals`/`run_benchmark`/`SimpleGCN`/`energy_distance`/`_sub`/`get_metric` 均为现有符号(已在本仓核对)。

---

## Execution Handoff

(由上层 skill 决定执行方式;若问到偏好:推荐 subagent-driven,每个 task 派一个 Opus 子代理实现 + 两段式审查。)
