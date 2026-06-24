import pytest


def test_squidpy_importable():
    """squidpy 是 niche 建图的新硬依赖,必须能 import 且暴露 gr.spatial_neighbors。"""
    sq = pytest.importorskip("squidpy")
    assert hasattr(sq, "gr"), "squidpy.gr 模块缺失"
    assert hasattr(sq.gr, "spatial_neighbors"), "squidpy.gr.spatial_neighbors 缺失"


def test_anndata_importable():
    """squidpy 走 AnnData 容器,anndata 必须随之可用。"""
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(X=np.zeros((3, 2), dtype=float))
    a.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert a.obsm["spatial"].shape == (3, 2)


def test_pyproject_declares_squidpy():
    """pyproject 必须把 squidpy 写进 dependencies(防止只在本机 pip 装而漏声明)。"""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    assert "squidpy" in text, "pyproject.toml 未声明 squidpy 依赖"
