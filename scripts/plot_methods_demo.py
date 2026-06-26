"""Demo: run the benchmark on synthetic data and save the seed/niche method-comparison plot.
Real datasets: replace make_synthetic with an Adapter().load() and a real perturbation list."""
import argparse
import matplotlib
matplotlib.use("Agg")
from spbench.synthetic import make_synthetic
from spbench.config import run_benchmark
from spbench.plotting import plot_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_demo.png")
    args = ap.parse_args()
    data = make_synthetic(seed=0)
    res = run_benchmark(data, perturbations=["P0"], gcn_kwargs={"hidden": 32, "epochs": 20},
                        progress=False)
    fig = plot_delta(res)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
